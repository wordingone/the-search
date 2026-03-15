"""
Export trained propagator weights to WebGPU-compatible format.

Converts PyTorch state dict to a binary format that can be loaded in WebGPU.

Usage:
    python scripts/export_weights.py --input weights/propagator.pt --output mirror/public/weights.bin
"""

import argparse
import struct
import numpy as np
import torch
from pathlib import Path
import json


def export_for_webgpu(state_dict: dict, output_path: str):
    """
    Export PyTorch state dict to WebGPU-compatible binary format.

    Format:
        - Header: magic number (4 bytes), num_tensors (4 bytes)
        - For each tensor:
            - name_length (4 bytes)
            - name (variable)
            - num_dims (4 bytes)
            - shape (num_dims * 4 bytes)
            - data_type (4 bytes): 0=float32, 1=float16
            - data (shape.prod() * sizeof(dtype) bytes)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Metadata for debugging
    metadata = {
        'tensors': [],
        'total_bytes': 0,
        'total_parameters': 0,
    }

    with open(output_path, 'wb') as f:
        # Write header
        f.write(struct.pack('I', 0x4D495252))  # Magic: "MIRR"
        f.write(struct.pack('I', len(state_dict)))

        for name, tensor in state_dict.items():
            # Convert to numpy float32
            arr = tensor.cpu().numpy().astype(np.float32)

            # Write name
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('I', len(name_bytes)))
            f.write(name_bytes)

            # Write shape
            f.write(struct.pack('I', len(arr.shape)))
            for dim in arr.shape:
                f.write(struct.pack('I', dim))

            # Write data type (0 = float32)
            f.write(struct.pack('I', 0))

            # Write data
            f.write(arr.tobytes())

            # Update metadata
            metadata['tensors'].append({
                'name': name,
                'shape': list(arr.shape),
                'size_bytes': arr.nbytes,
                'num_params': arr.size,
            })
            metadata['total_bytes'] += arr.nbytes
            metadata['total_parameters'] += arr.size

    # Write metadata JSON
    meta_path = output_path.with_suffix('.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Exported {len(state_dict)} tensors")
    print(f"Total parameters: {metadata['total_parameters']:,}")
    print(f"Total size: {metadata['total_bytes']:,} bytes ({metadata['total_bytes']/1024/1024:.2f} MB)")
    print(f"Binary output: {output_path}")
    print(f"Metadata output: {meta_path}")

    return metadata


def validate_export(weights_path: str, state_dict: dict):
    """
    Validate that exported weights can be read back correctly.
    """
    with open(weights_path, 'rb') as f:
        # Read header
        magic = struct.unpack('I', f.read(4))[0]
        assert magic == 0x4D495252, f"Invalid magic number: {hex(magic)}"

        num_tensors = struct.unpack('I', f.read(4))[0]
        assert num_tensors == len(state_dict), f"Tensor count mismatch: {num_tensors} vs {len(state_dict)}"

        for name, expected_tensor in state_dict.items():
            # Read name
            name_len = struct.unpack('I', f.read(4))[0]
            read_name = f.read(name_len).decode('utf-8')

            # Read shape
            num_dims = struct.unpack('I', f.read(4))[0]
            shape = tuple(struct.unpack('I', f.read(4))[0] for _ in range(num_dims))

            # Read data type
            dtype = struct.unpack('I', f.read(4))[0]
            assert dtype == 0, f"Unexpected dtype: {dtype}"

            # Read data
            num_elements = np.prod(shape)
            data = np.frombuffer(f.read(num_elements * 4), dtype=np.float32)
            data = data.reshape(shape)

            # Validate
            expected = expected_tensor.cpu().numpy().astype(np.float32)
            if not np.allclose(data, expected, rtol=1e-5, atol=1e-5):
                print(f"WARNING: Mismatch in tensor {read_name}")
                print(f"  Max diff: {np.abs(data - expected).max()}")
            else:
                print(f"  Validated: {read_name} {shape}")

    print("\nValidation complete!")


def main():
    parser = argparse.ArgumentParser(description='Export propagator weights for WebGPU')
    parser.add_argument('--input', type=str, required=True,
                        help='Input PyTorch weights file (.pt)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output binary weights file (.bin)')
    parser.add_argument('--validate', action='store_true',
                        help='Validate exported weights')

    args = parser.parse_args()

    # Load weights
    print(f"Loading weights from: {args.input}")
    state_dict = torch.load(args.input, map_location='cpu')

    # Handle checkpoint format (with optimizer state, etc.)
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']

    print(f"Found {len(state_dict)} tensors")

    # Export
    export_for_webgpu(state_dict, args.output)

    # Validate if requested
    if args.validate:
        print("\nValidating export...")
        validate_export(args.output, state_dict)


if __name__ == '__main__':
    main()
