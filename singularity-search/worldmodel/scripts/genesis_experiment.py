#!/usr/bin/env python
"""
Genesis Training Experiment Script

Deterministic, parameterized training script for Genesis experiments.
Designed to be invoked by /genesis-train skill with minimal token overhead.

Usage:
    python scripts/genesis_experiment.py --mode train --iterations 500 --eval-interval 100
    python scripts/genesis_experiment.py --mode evaluate --checkpoint path/to/checkpoint.pt
    python scripts/genesis_experiment.py --mode profile --iterations 10

Outputs structured JSON to stdout for easy parsing by orchestration layer.
"""

import sys
from pathlib import Path

# Add project root to PYTHONPATH
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description='Genesis Training Experiment')

    # Mode
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'evaluate', 'profile', 'resume'],
                        help='Experiment mode')

    # Training parameters
    parser.add_argument('--iterations', type=int, default=500,
                        help='Number of training iterations')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size (default: 1 for memory efficiency)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--eval-interval', type=int, default=100,
                        help='Evaluate every N iterations')
    parser.add_argument('--save-interval', type=int, default=500,
                        help='Save checkpoint every N iterations')

    # Loss configuration
    parser.add_argument('--use-perceptual', action='store_true',
                        help='Use perceptual loss')
    parser.add_argument('--perceptual-weight', type=float, default=0.1,
                        help='Weight for perceptual loss')

    # Tokenizer configuration
    parser.add_argument('--load-tokenizer', type=str, default=None,
                        help='Path to pre-trained tokenizer checkpoint')
    parser.add_argument('--freeze-tokenizer', action='store_true',
                        help='Freeze tokenizer weights')
    parser.add_argument('--use-fsq', action='store_true',
                        help='Enable FSQ discrete quantization (prevents slot corruption)')

    # Checkpoint/Resume
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint path (for evaluate/resume modes)')
    parser.add_argument('--save-dir', type=str, default='checkpoints/genesis_experiment',
                        help='Directory to save checkpoints')

    # Data configuration
    parser.add_argument('--data-mode', type=str, required=True,
                        choices=['synthetic', 'jat', 'webvid', 'local', 'openvid', 'panda70m', 'finevideo'],
                        help='Data source (REQUIRED). openvid=720p native (PRIMARY), finevideo=720p fast streaming, panda70m=720p alt, jat=256p legacy, webvid=diverse, local=custom, synthetic=validation only')
    parser.add_argument('--image-size', type=int, default=256,
                        help='Image size')
    parser.add_argument('--seq-length', type=int, default=8,
                        help='Sequence length')

    # Evaluation
    parser.add_argument('--eval-samples', type=int, default=16,
                        help='Number of samples for evaluation')

    # Meta-optimization
    parser.add_argument('--meta-file', type=str,
                        default='scripts/genesis_meta.json',
                        help='Path to meta-optimization tracking file')
    parser.add_argument('--experiment-name', type=str, default=None,
                        help='Name for this experiment (auto-generated if not provided)')

    # Video cache (eliminates WebVid HTTP bottleneck)
    parser.add_argument('--video-cache-dir', type=str, default='data/video_cache',
                        help='Directory for persistent video cache (default: data/video_cache). '
                             'First run downloads and caches; subsequent runs load from disk (~300x faster)')
    parser.add_argument('--video-cache-size', type=int, default=50,
                        help='Number of video batches to cache (default: 50). '
                             'Each batch ~50MB at 720p. Total ~2-3GB for 50 batches')
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable video caching (force fresh HTTP downloads every iteration)')

    # Output
    parser.add_argument('--quiet', action='store_true',
                        help='Minimal output (JSON only)')
    parser.add_argument('--output-json', type=str, default=None,
                        help='Write results to JSON file')

    # GPU resource management
    parser.add_argument('--gpu-cap', type=float, default=1.0,
                        help='GPU memory fraction cap (0.0-1.0, default: 1.0). '
                             'Note: 720p needs ~100%% VRAM, only reduce if using gradient checkpointing')
    parser.add_argument('--gpu-yield-ms', type=int, default=10,
                        help='Milliseconds to yield GPU between iterations (default: 10). '
                             'Prevents 100%% GPU lock that makes workstation unresponsive. 0=disabled')
    parser.add_argument('--gradient-checkpoint', action='store_true',
                        help='Enable gradient checkpointing to reduce peak VRAM ~30%% '
                             '(trades ~15%% speed). Required for --gpu-cap 0.9 at 720p')

    return parser.parse_args()


def load_model(args, device):
    """Load Genesis model with configuration."""
    from genesis.genesis_model import Genesis, GenesisConfig

    config = GenesisConfig(
        image_size=args.image_size,
        use_fsq=getattr(args, 'use_fsq', False),  # FSQ prevents slot corruption
        gradient_checkpoint=getattr(args, 'gradient_checkpoint', False),
    )
    model = Genesis(config).to(device)

    # Load pre-trained tokenizer if specified
    if args.load_tokenizer and os.path.exists(args.load_tokenizer):
        tok_ckpt = torch.load(args.load_tokenizer, map_location=device, weights_only=False)
        # Load with strict=False to handle FSQ layers not in pre-trained tokenizer
        missing, unexpected = model.tokenizer.load_state_dict(
            tok_ckpt['tokenizer_state_dict'],
            strict=False
        )
        if not args.quiet:
            print(f'Loaded tokenizer from {args.load_tokenizer}', file=sys.stderr)
            if missing:
                print(f'  FSQ layers initialized randomly: {len(missing)} keys', file=sys.stderr)

    # Freeze tokenizer if specified
    if args.freeze_tokenizer:
        for name, p in model.tokenizer.named_parameters():
            # Keep FSQ projection layers trainable even when tokenizer is frozen
            if 'fsq_proj' in name:
                p.requires_grad = True
            else:
                p.requires_grad = False
        if not args.quiet:
            fsq_params = sum(p.numel() for n, p in model.tokenizer.named_parameters() if 'fsq_proj' in n)
            print(f'Tokenizer frozen (FSQ layers trainable: {fsq_params} params)', file=sys.stderr)

    return model, config


def load_perceptual_loss(args, device):
    """Load perceptual loss function if enabled."""
    if not args.use_perceptual:
        return None

    from genesis.tokenizer.losses import VGGPerceptualLoss
    perceptual_fn = VGGPerceptualLoss().to(device)

    # Freeze perceptual network
    for p in perceptual_fn.parameters():
        p.requires_grad = False

    return perceptual_fn


def create_data_iterator(args):
    """Create data iterator based on data mode."""
    if args.data_mode == 'synthetic':
        # Synthetic data - infinite random noise (only for specific validation)
        return None  # Handled specially in get_data_batch

    elif args.data_mode == 'jat':
        from genesis.pilot.stream_hf import JATStreamDataset
        dataset = JATStreamDataset(
            game="atari-breakout",
            seq_length=args.seq_length,
            image_size=args.image_size,
            shuffle=True,
            buffer_size=500,
        )
        return iter(dataset)

    elif args.data_mode == 'webvid':
        use_cache = not getattr(args, 'no_cache', False)
        cache_dir = getattr(args, 'video_cache_dir', 'data/video_cache')
        cache_size = getattr(args, 'video_cache_size', 50)

        if use_cache:
            from genesis.pilot.stream_hf import CachedWebVidDataset
            dataset = CachedWebVidDataset(
                cache_dir=cache_dir,
                max_videos=cache_size,
                url_source="webvid",
                seq_length=args.seq_length,
                image_size=args.image_size,
                shuffle=True,
                buffer_size=100,
            )
        else:
            from genesis.pilot.stream_hf import URLVideoStreamDataset
            dataset = URLVideoStreamDataset(
                url_source="webvid",
                seq_length=args.seq_length,
                image_size=args.image_size,
                shuffle=True,
                buffer_size=100,
            )
        return iter(dataset)

    elif args.data_mode == 'local':
        from genesis.pilot.stream_hf import LocalVideoDataset
        dataset = LocalVideoDataset(
            video_dir="data/videos",
            seq_length=args.seq_length,
            image_size=args.image_size,
        )
        return iter(dataset)

    elif args.data_mode == 'openvid':
        from genesis.data import OpenVidStreamDataset
        dataset = OpenVidStreamDataset(
            seq_length=args.seq_length,
            image_size=args.image_size,
            shuffle=True,
            buffer_size=100,
        )
        return iter(dataset)

    elif args.data_mode == 'panda70m':
        from genesis.data import Panda70MStreamDataset
        dataset = Panda70MStreamDataset(
            seq_length=args.seq_length,
            image_size=args.image_size,
            shuffle=True,
            buffer_size=100,
        )
        return iter(dataset)

    elif args.data_mode == 'finevideo':
        from genesis.data.openvid import FineVideoStreamDataset
        dataset = FineVideoStreamDataset(
            seq_length=args.seq_length,
            image_size=args.image_size,
            shuffle=True,
            buffer_size=100,
        )
        return iter(dataset)

    else:
        raise ValueError(f'Unknown data mode: {args.data_mode}')


# Global data iterator (reused across batches)
_data_iterator = None


def get_data_batch(args, device, data_iter=None):
    """Get a batch of data based on data mode."""
    global _data_iterator

    if args.data_mode == 'synthetic':
        # Fast synthetic data - no I/O overhead (only for specific validation)
        return torch.randn(args.batch_size, args.seq_length, 3,
                          args.image_size, args.image_size, device=device)

    # Use provided iterator or global one
    if data_iter is not None:
        iterator = data_iter
    else:
        if _data_iterator is None:
            _data_iterator = create_data_iterator(args)
        iterator = _data_iterator

    # Collect batch
    frames_list = []
    for _ in range(args.batch_size):
        try:
            sample = next(iterator)
            frames = sample['frames']  # [T, C, H, W]
            # Normalize to [0, 1] (tokenizer decoder uses Sigmoid -> [0, 1] output)
            if frames.max() > 1.0:
                frames = frames / 255.0
            frames_list.append(frames)
        except StopIteration:
            # Recreate iterator if exhausted
            if data_iter is None:
                _data_iterator = create_data_iterator(args)
                iterator = _data_iterator
            sample = next(iterator)
            frames = sample['frames']
            if frames.max() > 1.0:
                frames = frames / 255.0
            # Bug #2 Fix: No normalization - keep in [0, 1] range
            # frames = frames * 2.0 - 1.0  # REMOVED
            frames_list.append(frames)

    batch = torch.stack(frames_list).to(device)  # [B, T, C, H, W]
    return batch


def compute_loss(model, perceptual_fn, batch, args):
    """Compute training loss.

    Model.forward() now includes tokenizer roundtrip loss (tok_recon_loss)
    in its total_loss. We add optional perceptual loss on top.

    Perceptual loss is applied to BOTH:
    1. Dynamics predicted video (pred_video vs target) - forces dynamics to predict detail
    2. Tokenizer roundtrip (tok_recon vs original) - forces tokenizer to preserve color/detail
    """
    out = model(batch)

    pred_video = out['pred_video']
    target_video = batch[:, 1:]

    if perceptual_fn is not None:
        B, T, C, H, W = pred_video.shape
        # Perceptual loss on dynamics prediction
        perceptual_loss = perceptual_fn(
            pred_video.reshape(B * T, C, H, W),
            target_video.reshape(B * T, C, H, W)
        )
        total_loss = out['total_loss'] + args.perceptual_weight * perceptual_loss
    else:
        perceptual_loss = torch.tensor(0.0)
        total_loss = out['total_loss']

    return {
        'total': total_loss,
        'mse': out['recon_loss'],
        'perceptual': perceptual_loss,
    }


def evaluate_quality(model, args, device, config=None):
    """Evaluate generation quality with CLIP-IQA."""
    model.eval()
    model.dynamics.slot_norm_mode = 'clip'

    # Get action_dim from config or infer from model
    if config is not None:
        action_dim = config.action_dim
    else:
        # Infer from action_encoder input layer
        action_dim = model.action_encoder[0].in_features

    scores = []

    try:
        import pyiqa
        clip_iqa = pyiqa.create_metric('clipiqa', device=device)
    except Exception as e:
        return {'clip_iqa': None, 'error': str(e)}

    with torch.no_grad():
        for i in range(args.eval_samples):
            # Generate from random seed in [0, 1] (matches tokenizer's Sigmoid output range)
            seed = torch.rand(1, 2, 3, args.image_size, args.image_size, device=device)
            actions = torch.zeros(1, args.seq_length, action_dim, device=device)

            try:
                frames = model.generate(seed, actions, num_frames=args.seq_length)
                middle_frame = frames[0, args.seq_length // 2]

                # Already in [0, 1] from tokenizer Sigmoid output
                frame_normalized = middle_frame.unsqueeze(0).clamp(0, 1)
                score = clip_iqa(frame_normalized).item()
                scores.append(score)
            except Exception as e:
                continue

    model.train()

    if scores:
        return {
            'clip_iqa': float(np.mean(scores)),
            'clip_iqa_std': float(np.std(scores)),
            'clip_iqa_min': float(min(scores)),
            'clip_iqa_max': float(max(scores)),
            'num_samples': len(scores)
        }
    else:
        return {'clip_iqa': None, 'error': 'No valid samples generated'}


def train(args, start_iteration=0, checkpoint_state=None):
    """Main training loop."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # GPU resource cap: limit VRAM usage to allow other processes
    if device.type == 'cuda' and hasattr(args, 'gpu_cap') and args.gpu_cap < 1.0:
        torch.cuda.set_per_process_memory_fraction(args.gpu_cap, 0)
        if not getattr(args, 'quiet', False):
            total_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2
            cap_mb = total_mb * args.gpu_cap
            print(f'GPU memory cap: {args.gpu_cap*100:.0f}% ({cap_mb:.0f}/{total_mb:.0f} MB)',
                  file=sys.stderr)

    # Load model and loss
    model, config = load_model(args, device)
    perceptual_fn = load_perceptual_loss(args, device)

    # Setup optimizer (only trainable params)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    # Restore checkpoint state if resuming
    if checkpoint_state is not None:
        model.load_state_dict(checkpoint_state['model_state_dict'])
        optimizer.load_state_dict(checkpoint_state['optimizer_state_dict'])
        losses = checkpoint_state.get('losses', {'total': [], 'mse': [], 'perceptual': []})
        eval_results = checkpoint_state.get('eval_results', [])
        if not args.quiet:
            print(f'Resumed from iteration {start_iteration}', file=sys.stderr)
    else:
        losses = {'total': [], 'mse': [], 'perceptual': []}
        eval_results = []

    # Setup save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Training state
    start_time = time.time()
    iteration_times = []

    # Count parameters
    trainable_count = sum(p.numel() for p in trainable_params)
    total_count = sum(p.numel() for p in model.parameters())

    if not args.quiet:
        print(f'Training {trainable_count:,} / {total_count:,} parameters', file=sys.stderr)
        print(f'Perceptual loss: {args.use_perceptual}', file=sys.stderr)
        if start_iteration > 0:
            print(f'Continuing training from iteration {start_iteration}', file=sys.stderr)

    # Training loop
    for i in range(start_iteration, start_iteration + args.iterations):
        iter_start = time.time()

        # Get data
        batch = get_data_batch(args, device)

        # Forward + backward
        optimizer.zero_grad()
        loss_dict = compute_loss(model, perceptual_fn, batch, args)
        loss_dict['total'].backward()
        optimizer.step()

        # GPU yield: sync and sleep to give other processes GPU time
        gpu_yield_ms = getattr(args, 'gpu_yield_ms', 10)
        if device.type == 'cuda' and gpu_yield_ms > 0:
            torch.cuda.synchronize()
            time.sleep(gpu_yield_ms / 1000.0)

        # Periodic cache cleanup to reduce memory fragmentation
        if device.type == 'cuda' and (i + 1) % 50 == 0:
            torch.cuda.empty_cache()

        # Record
        iter_time = time.time() - iter_start
        iteration_times.append(iter_time)

        for k, v in loss_dict.items():
            losses[k].append(v.item())

        # Progress output
        if not args.quiet and (i + 1) % 100 == 0:
            avg_time = np.mean(iteration_times[-100:])
            total_iters = start_iteration + args.iterations
            print(f'Iter {i+1}/{total_iters}: loss={loss_dict["total"].item():.4f} '
                  f'mse={loss_dict["mse"].item():.4f} perc={loss_dict["perceptual"].item():.4f} '
                  f'({avg_time*1000:.0f}ms/iter)', file=sys.stderr)

        # Evaluation
        if (i + 1) % args.eval_interval == 0:
            eval_result = evaluate_quality(model, args, device, config)
            eval_result['iteration'] = i + 1
            eval_result['elapsed_seconds'] = time.time() - start_time
            eval_results.append(eval_result)

            if not args.quiet:
                clip_iqa = eval_result.get('clip_iqa', 'N/A')
                print(f'  Eval @ {i+1}: CLIP-IQA={clip_iqa}', file=sys.stderr)

        # Save checkpoint
        if (i + 1) % args.save_interval == 0:
            ckpt_path = os.path.join(args.save_dir, f'checkpoint_{i+1}.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'iteration': i + 1,
                'losses': losses,
                'eval_results': eval_results,
                'args': vars(args),
            }, ckpt_path)

    # Final evaluation
    final_iteration = start_iteration + args.iterations
    final_eval = evaluate_quality(model, args, device, config)
    final_eval['iteration'] = final_iteration
    final_eval['elapsed_seconds'] = time.time() - start_time
    eval_results.append(final_eval)

    # Save final checkpoint
    final_ckpt_path = os.path.join(args.save_dir, 'final.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': final_iteration,
        'losses': losses,
        'eval_results': eval_results,
        'args': vars(args),
    }, final_ckpt_path)

    total_time = time.time() - start_time
    avg_iter_time = np.mean(iteration_times)

    # Build result
    result = {
        'status': 'completed',
        'iterations': final_iteration,
        'iterations_trained': args.iterations,
        'start_iteration': start_iteration,
        'total_time_seconds': total_time,
        'avg_iter_ms': avg_iter_time * 1000,
        'throughput_iters_per_sec': 1.0 / avg_iter_time,
        'loss_initial': losses['total'][0] if losses['total'] else None,
        'loss_final': losses['total'][-1] if losses['total'] else None,
        'loss_improvement_pct': (losses['total'][0] - losses['total'][-1]) / losses['total'][0] * 100 if losses['total'] else None,
        'final_clip_iqa': final_eval.get('clip_iqa'),
        'eval_history': eval_results,
        'checkpoint_path': final_ckpt_path,
        'trainable_params': trainable_count,
        'total_params': total_count,
        'config': {
            'use_perceptual': args.use_perceptual,
            'perceptual_weight': args.perceptual_weight,
            'freeze_tokenizer': args.freeze_tokenizer,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'data_mode': args.data_mode,
        }
    }

    # Update meta-optimization tracking
    update_meta(args, result)

    return result


def resume(args):
    """Resume training from checkpoint."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not args.checkpoint or not os.path.exists(args.checkpoint):
        return {'status': 'error', 'error': f'Checkpoint not found: {args.checkpoint}'}

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Extract start iteration
    start_iteration = ckpt.get('iteration', 0)

    if not args.quiet:
        print(f'Loading checkpoint from {args.checkpoint}', file=sys.stderr)
        print(f'Resuming from iteration {start_iteration}', file=sys.stderr)
        print(f'Will train for {args.iterations} more iterations (target: {start_iteration + args.iterations})', file=sys.stderr)

    # Call train with checkpoint state
    return train(args, start_iteration=start_iteration, checkpoint_state=ckpt)


def profile(args):
    """Profile training speed without full training."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # GPU resource cap
    if device.type == 'cuda' and hasattr(args, 'gpu_cap') and args.gpu_cap < 1.0:
        torch.cuda.set_per_process_memory_fraction(args.gpu_cap, 0)

    model, config = load_model(args, device)
    perceptual_fn = load_perceptual_loss(args, device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    # Warmup
    for _ in range(3):
        batch = get_data_batch(args, device)
        optimizer.zero_grad()
        loss_dict = compute_loss(model, perceptual_fn, batch, args)
        loss_dict['total'].backward()
        optimizer.step()

    # Profile
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    times = []

    for _ in range(args.iterations):
        start = time.time()
        batch = get_data_batch(args, device)
        optimizer.zero_grad()
        loss_dict = compute_loss(model, perceptual_fn, batch, args)
        loss_dict['total'].backward()
        optimizer.step()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        times.append(time.time() - start)

    avg_time = np.mean(times)
    std_time = np.std(times)

    return {
        'status': 'profiled',
        'iterations': args.iterations,
        'avg_iter_ms': avg_time * 1000,
        'std_iter_ms': std_time * 1000,
        'throughput_iters_per_sec': 1.0 / avg_time,
        'estimated_time_1000_iters': avg_time * 1000,
        'estimated_time_5000_iters': avg_time * 5000,
        'trainable_params': sum(p.numel() for p in trainable_params),
        'config': {
            'use_perceptual': args.use_perceptual,
            'batch_size': args.batch_size,
            'data_mode': args.data_mode,
        }
    }


def evaluate(args):
    """Evaluate a checkpoint."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not args.checkpoint or not os.path.exists(args.checkpoint):
        return {'status': 'error', 'error': f'Checkpoint not found: {args.checkpoint}'}

    model, config = load_model(args, device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    result = evaluate_quality(model, args, device, config)
    result['status'] = 'evaluated'
    result['checkpoint'] = args.checkpoint
    result['iteration'] = ckpt.get('iteration', 'unknown')

    return result


def update_meta(args, result):
    """Update meta-optimization tracking file."""
    meta_path = Path(args.meta_file)

    # Load existing meta
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            meta = json.load(f)
    else:
        meta = {
            'experiments': [],
            'best_efficiency': None,
            'optimization_suggestions': [],
            'total_training_time': 0,
            'total_iterations': 0,
        }

    # Create experiment record
    experiment = {
        'timestamp': datetime.now().isoformat(),
        'name': args.experiment_name or f'exp_{len(meta["experiments"]) + 1}',
        'iterations': result['iterations'],
        'time_seconds': result['total_time_seconds'],
        'final_clip_iqa': result.get('final_clip_iqa'),
        'loss_improvement_pct': result.get('loss_improvement_pct'),
        'avg_iter_ms': result['avg_iter_ms'],
        'config': result['config'],
    }

    # Compute efficiency metric: quality_gain / time
    if result.get('final_clip_iqa') is not None:
        baseline_iqa = 0.151  # MSE-only baseline
        quality_gain = max(0, result['final_clip_iqa'] - baseline_iqa)
        efficiency = quality_gain / result['total_time_seconds'] * 60  # per minute
        experiment['efficiency'] = efficiency

        # Update best efficiency
        if meta['best_efficiency'] is None or efficiency > meta['best_efficiency']['efficiency']:
            meta['best_efficiency'] = {
                'efficiency': efficiency,
                'config': result['config'],
                'experiment': experiment['name'],
            }

    meta['experiments'].append(experiment)
    meta['total_training_time'] += result['total_time_seconds']
    meta['total_iterations'] += result['iterations']

    # Generate optimization suggestions based on history
    meta['optimization_suggestions'] = generate_suggestions(meta)

    # Save
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)


def generate_suggestions(meta):
    """Generate optimization suggestions based on experiment history."""
    suggestions = []

    if len(meta['experiments']) < 2:
        suggestions.append('Run more experiments to gather optimization data')
        return suggestions

    # Analyze experiments
    exps = meta['experiments']

    # Check data mode impact
    synthetic_exps = [e for e in exps if e['config'].get('data_mode') == 'synthetic']
    real_exps = [e for e in exps if e['config'].get('data_mode') in ['jat', 'webvid', 'local']]

    if synthetic_exps:
        best_synthetic_iqa = max([e['final_clip_iqa'] for e in synthetic_exps if e['final_clip_iqa']] or [0])
        if best_synthetic_iqa > 0 and best_synthetic_iqa < 0.25:
            suggestions.append(f'Synthetic data ceiling at {best_synthetic_iqa:.3f} - use real data (jat/webvid)')

    if real_exps and synthetic_exps:
        avg_real_iqa = np.mean([e['final_clip_iqa'] for e in real_exps if e['final_clip_iqa']] or [0])
        avg_synth_iqa = np.mean([e['final_clip_iqa'] for e in synthetic_exps if e['final_clip_iqa']] or [0])
        if avg_real_iqa > avg_synth_iqa:
            suggestions.append(f'Real data improves quality by {(avg_real_iqa - avg_synth_iqa):.3f} over synthetic')

    # Check if perceptual loss helps
    perc_exps = [e for e in exps if e['config'].get('use_perceptual')]
    no_perc_exps = [e for e in exps if not e['config'].get('use_perceptual')]

    if perc_exps and no_perc_exps:
        perc_iqa = [e['final_clip_iqa'] for e in perc_exps if e['final_clip_iqa']]
        no_perc_iqa = [e['final_clip_iqa'] for e in no_perc_exps if e['final_clip_iqa']]
        if perc_iqa and no_perc_iqa:
            avg_perc_iqa = np.mean(perc_iqa)
            avg_no_perc_iqa = np.mean(no_perc_iqa)
            if avg_perc_iqa > avg_no_perc_iqa:
                suggestions.append(f'Perceptual loss improves quality by {(avg_perc_iqa - avg_no_perc_iqa):.3f}')

    # Check iteration efficiency
    recent = exps[-3:] if len(exps) >= 3 else exps
    efficiencies = [e.get('efficiency', 0) for e in recent if e.get('efficiency')]
    if efficiencies:
        avg_efficiency = np.mean(efficiencies)
        suggestions.append(f'Average efficiency: {avg_efficiency:.4f} quality/min')

    # Recommend best data mode based on history
    if real_exps:
        jat_exps = [e for e in real_exps if e['config'].get('data_mode') == 'jat']
        webvid_exps = [e for e in real_exps if e['config'].get('data_mode') == 'webvid']
        if jat_exps and webvid_exps:
            jat_iqa = np.mean([e['final_clip_iqa'] for e in jat_exps if e['final_clip_iqa']] or [0])
            webvid_iqa = np.mean([e['final_clip_iqa'] for e in webvid_exps if e['final_clip_iqa']] or [0])
            best_mode = 'jat' if jat_iqa >= webvid_iqa else 'webvid'
            suggestions.append(f'Best data mode: {best_mode}')

    # Suggest batch size adjustment based on memory
    suggestions.append('Consider batch_size=2 if memory allows for faster convergence')

    return suggestions


def main():
    args = get_args()

    if args.mode == 'train':
        result = train(args)
    elif args.mode == 'profile':
        result = profile(args)
    elif args.mode == 'evaluate':
        result = evaluate(args)
    elif args.mode == 'resume':
        result = resume(args)
    else:
        result = {'status': 'error', 'error': f'Unknown mode: {args.mode}'}

    # Output JSON result
    output = json.dumps(result, indent=2)
    print(output)

    if args.output_json:
        with open(args.output_json, 'w') as f:
            f.write(output)


if __name__ == '__main__':
    main()
