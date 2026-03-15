#!/usr/bin/env python3
"""Genesis Debug Capture - Deterministic symptom capture for debugging.

Usage:
    python scripts/genesis_debug_capture.py \
        --checkpoint checkpoints/genesis_720_extended/checkpoint_5000_with_config.pt \
        --seed-source webvid \
        --frames 50 \
        --output outputs/debug/

Output:
    outputs/debug/symptom_capture.json - Numeric failure data
    outputs/debug/trajectory_plot.png - Visual trajectory of metrics
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_model(checkpoint_path: str, device: str, slot_norm_mode: str = 'clip'):
    """Load Genesis model with specified configuration."""
    from genesis.genesis_model import Genesis

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt['config']

    # Override slot_norm_mode for inference
    original_mode = getattr(config, 'slot_norm_mode', 'decay')
    config.slot_norm_mode = slot_norm_mode
    print(f"Slot norm mode: {original_mode} -> {slot_norm_mode}")

    model = Genesis(config).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    return model, config


def load_seed_video(
    source: str,
    config,
    device: str,
    num_frames: int = 8,
) -> Tuple[torch.Tensor, str]:
    """Load seed video from specified source.

    Args:
        source: One of 'webvid', 'jat', 'random', or 'file:/path/to/video.mp4'
        config: Model config
        device: torch device
        num_frames: Number of seed frames to use

    Returns:
        seed_video: [1, T, C, H, W] tensor
        source_description: Human-readable description
    """
    if source == 'random':
        # Random noise seed (useful for isolating encoder issues)
        seed = torch.rand(1, num_frames, 3, config.image_size, config.image_size, device=device)
        return seed, "random_noise"

    elif source.startswith('file:'):
        # Load from file
        import imageio
        video_path = source[5:]
        reader = imageio.get_reader(video_path)
        frames = [reader.get_data(i) for i in range(min(num_frames, reader.count_frames()))]
        reader.close()

        seed = torch.stack([
            torch.from_numpy(f).permute(2, 0, 1).float() / 255.0
            for f in frames
        ]).unsqueeze(0).to(device)

        # Resize if needed
        if seed.shape[-1] != config.image_size:
            seed = F.interpolate(
                seed.view(-1, 3, seed.shape[-2], seed.shape[-1]),
                size=(config.image_size, config.image_size),
                mode='bilinear',
                align_corners=False,
            ).view(1, -1, 3, config.image_size, config.image_size)

        return seed, f"file:{video_path}"

    else:
        # Dataset source (webvid, jat, etc.)
        from genesis.pilot.stream_hf import create_streaming_loader

        loader_kwargs = {
            'dataset_name': source,
            'batch_size': 1,
            'seq_length': num_frames,
            'image_size': config.image_size,
            'shuffle': False,  # Deterministic
        }
        if source == 'jat':
            loader_kwargs['game'] = 'atari-breakout'

        try:
            loader = create_streaming_loader(**loader_kwargs)
            batch = next(iter(loader))
            video = batch.get('frames', batch.get('video'))
            return video.to(device), f"dataset:{source}"
        except Exception as e:
            print(f"Failed to load from {source}: {e}")
            print("Falling back to random seed")
            seed = torch.rand(1, num_frames, 3, config.image_size, config.image_size, device=device)
            return seed, f"random_fallback (original: {source})"


def capture_generation_trajectory(
    model,
    config,
    seed_video: torch.Tensor,
    num_frames: int,
    device: str,
) -> Dict:
    """Capture detailed metrics during generation.

    Returns dict with:
        - Per-frame metrics (mean, std, min, max)
        - Per-frame latent metrics
        - Per-frame slot metrics
        - Failure detection
    """
    model.reset_state()

    # Encode seed
    context_latents, encoder_output = model.encode(seed_video)
    current_context = context_latents

    B, T_seed, C, H, W = seed_video.shape

    # Initial metrics
    results = {
        'seed': {
            'shape': list(seed_video.shape),
            'mean': seed_video.mean().item(),
            'std': seed_video.std().item(),
            'min': seed_video.min().item(),
            'max': seed_video.max().item(),
        },
        'encoded_context': {
            'shape': list(context_latents.shape),
            'mean': context_latents.mean().item(),
            'std': context_latents.std().item(),
        },
        'config': {
            'image_size': config.image_size,
            'window_size': config.window_size,
            'slot_norm_mode': config.slot_norm_mode,
            'num_slots': getattr(config, 'num_slots', None),
            'slot_dim': getattr(config, 'slot_dim', None),
            'slot_decay': getattr(config, 'slot_decay', None),
        },
        'frames': [],
        'failure': None,
    }

    prev_slots = None
    prev_frame_tensor = None

    with torch.no_grad():
        for i in range(num_frames):
            action = torch.zeros(1, current_context.shape[1], config.action_dim, device=device)

            # Dynamics forward
            if config.use_slots:
                pred, prev_slots = model.dynamics(current_context, action, prev_slots)
            else:
                pred = model.dynamics(current_context, action)
                prev_slots = None

            next_latent = pred[:, -1:]

            # Decode
            next_frame = model.decode(next_latent)

            # Capture frame metrics
            frame_data = {
                'idx': i,
                'frame': {
                    'mean': next_frame.mean().item(),
                    'std': next_frame.std().item(),
                    'min': next_frame.min().item(),
                    'max': next_frame.max().item(),
                },
                'latent': {
                    'mean': next_latent.mean().item(),
                    'std': next_latent.std().item(),
                    'min': next_latent.min().item(),
                    'max': next_latent.max().item(),
                },
                'context': {
                    'length': current_context.shape[1],
                    'mean': current_context.mean().item(),
                    'std': current_context.std().item(),
                },
            }

            # Slot metrics
            if prev_slots is not None:
                frame_data['slots'] = {
                    'norm': prev_slots.norm().item(),
                    'mean': prev_slots.mean().item(),
                    'std': prev_slots.std().item(),
                    'max_abs': prev_slots.abs().max().item(),
                }

            # Frame-to-frame comparison
            if prev_frame_tensor is not None:
                frame_data['delta'] = {
                    'mse': F.mse_loss(next_frame, prev_frame_tensor).item(),
                    'cosine_sim': F.cosine_similarity(
                        next_frame.view(1, -1),
                        prev_frame_tensor.view(1, -1)
                    ).item(),
                }

            results['frames'].append(frame_data)

            # Failure detection
            if results['failure'] is None:
                if frame_data['frame']['mean'] < 0.05:
                    results['failure'] = {
                        'frame_idx': i,
                        'type': 'BLACK_FRAME',
                        'frame_mean': frame_data['frame']['mean'],
                    }
                elif frame_data['latent']['std'] < 0.01:
                    results['failure'] = {
                        'frame_idx': i,
                        'type': 'LATENT_COLLAPSE',
                        'latent_std': frame_data['latent']['std'],
                    }

            # Update for next iteration
            prev_frame_tensor = next_frame.clone()
            window = min(current_context.shape[1] + 1, config.window_size // 4)
            current_context = torch.cat([current_context, next_latent], dim=1)[:, -window:]

    # Summary statistics
    frame_means = [f['frame']['mean'] for f in results['frames']]
    latent_stds = [f['latent']['std'] for f in results['frames']]
    slot_norms = [f['slots']['norm'] for f in results['frames'] if 'slots' in f]

    results['summary'] = {
        'total_frames': num_frames,
        'frame_mean': {
            'initial': frame_means[0] if frame_means else None,
            'final': frame_means[-1] if frame_means else None,
            'min': min(frame_means) if frame_means else None,
            'max': max(frame_means) if frame_means else None,
        },
        'latent_std': {
            'initial': latent_stds[0] if latent_stds else None,
            'final': latent_stds[-1] if latent_stds else None,
            'collapse_detected': latent_stds[-1] < 0.1 * latent_stds[0] if latent_stds else False,
        },
        'slot_norm': {
            'initial': slot_norms[0] if slot_norms else None,
            'final': slot_norms[-1] if slot_norms else None,
            'ratio': slot_norms[-1] / slot_norms[0] if slot_norms and slot_norms[0] > 0 else None,
            'explosion_detected': slot_norms[-1] > 2 * slot_norms[0] if slot_norms else False,
        },
    }

    # Classify failure type
    if results['failure']:
        if results['summary']['latent_std']['collapse_detected']:
            results['failure']['root_hypothesis'] = 'LATENT_COLLAPSE'
        elif results['summary']['slot_norm']['explosion_detected']:
            results['failure']['root_hypothesis'] = 'SLOT_EXPLOSION'
        else:
            results['failure']['root_hypothesis'] = 'UNKNOWN'

    return results


def plot_trajectories(results: Dict, output_path: str):
    """Plot metric trajectories for visual inspection."""
    try:
        import matplotlib.pyplot as plt

        frames = results['frames']
        indices = [f['idx'] for f in frames]

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Frame mean
        ax = axes[0, 0]
        ax.plot(indices, [f['frame']['mean'] for f in frames], 'b-', label='mean')
        ax.axhline(y=0.05, color='r', linestyle='--', label='black threshold')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Frame Mean')
        ax.set_title('Frame Mean Trajectory')
        ax.legend()

        # Latent std
        ax = axes[0, 1]
        ax.plot(indices, [f['latent']['std'] for f in frames], 'g-')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Latent Std')
        ax.set_title('Latent Std Trajectory')
        ax.set_yscale('log')

        # Slot norm
        ax = axes[1, 0]
        slot_norms = [f['slots']['norm'] for f in frames if 'slots' in f]
        if slot_norms:
            ax.plot(range(len(slot_norms)), slot_norms, 'm-')
            ax.set_xlabel('Frame')
            ax.set_ylabel('Slot Norm')
            ax.set_title('Slot Norm Trajectory')

        # Context length
        ax = axes[1, 1]
        ax.plot(indices, [f['context']['length'] for f in frames], 'c-')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Context Length')
        ax.set_title('Context Window Length')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Saved trajectory plot: {output_path}")

    except ImportError:
        print("matplotlib not available, skipping plot")


def main():
    parser = argparse.ArgumentParser(description='Genesis Debug Capture')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--seed-source', default='webvid',
                        help='Seed source: webvid, jat, random, or file:/path')
    parser.add_argument('--frames', type=int, default=50,
                        help='Number of frames to generate')
    parser.add_argument('--seed-frames', type=int, default=8,
                        help='Number of seed frames to use')
    parser.add_argument('--output', default='outputs/debug/',
                        help='Output directory')
    parser.add_argument('--slot-norm-mode', default='clip',
                        choices=['decay', 'clip', 'layernorm', 'none'],
                        help='Slot normalization mode')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GENESIS DEBUG CAPTURE")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Seed source: {args.seed_source}")
    print(f"Frames to generate: {args.frames}")
    print(f"Device: {args.device}")
    print("=" * 70)

    # Load model
    model, config = load_model(args.checkpoint, args.device, args.slot_norm_mode)

    # Load seed
    print(f"\nLoading seed video from: {args.seed_source}")
    seed_video, source_desc = load_seed_video(
        args.seed_source, config, args.device, args.seed_frames
    )
    print(f"Seed video shape: {seed_video.shape}")
    print(f"Seed source: {source_desc}")

    # Capture trajectory
    print(f"\nCapturing generation trajectory ({args.frames} frames)...")
    results = capture_generation_trajectory(
        model, config, seed_video, args.frames, args.device
    )
    results['source'] = source_desc

    # Save results
    output_path = output_dir / 'symptom_capture.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_path}")

    # Plot trajectories
    plot_path = output_dir / 'trajectory_plot.png'
    plot_trajectories(results, str(plot_path))

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Seed: mean={results['seed']['mean']:.3f}, std={results['seed']['std']:.3f}")
    print(f"Frames generated: {results['summary']['total_frames']}")
    print(f"Frame mean: {results['summary']['frame_mean']['initial']:.4f} -> {results['summary']['frame_mean']['final']:.4f}")
    print(f"Latent std: {results['summary']['latent_std']['initial']:.4f} -> {results['summary']['latent_std']['final']:.4f}")

    if results['summary']['slot_norm']['initial']:
        ratio = results['summary']['slot_norm']['ratio']
        print(f"Slot norm: {results['summary']['slot_norm']['initial']:.2f} -> {results['summary']['slot_norm']['final']:.2f} ({ratio:.1f}x)")

    if results['failure']:
        print(f"\nFAILURE DETECTED at frame {results['failure']['frame_idx']}")
        print(f"  Type: {results['failure']['type']}")
        print(f"  Root hypothesis: {results['failure']['root_hypothesis']}")
    else:
        print("\nNo failure detected")

    print("=" * 70)


if __name__ == '__main__':
    main()
