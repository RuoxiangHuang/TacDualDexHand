"""Validate collected tactile shape dataset.

This script checks the integrity and quality of collected tactile data,
including class distribution, image statistics, and sample visualization.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import zarr


def validate_dataset(zarr_path: str, visualize: bool = True, output_dir: str | None = None):
    """Validate the collected tactile dataset.
    
    Args:
        zarr_path: Path to the zarr dataset directory.
        visualize: Whether to create visualization plots.
        output_dir: Directory to save visualization (default: same as dataset).
    """
    print(f"Loading dataset from: {zarr_path}")
    root = zarr.open(zarr_path, mode='r')
    
    # Load data
    tactile_images = root['tactile_images'][:]
    labels = root['labels'][:]
    shape_names = root['shape_names'][:]
    
    # Print dataset information
    print(f"\n{'='*60}")
    print(f"Dataset Information:")
    print(f"{'='*60}")
    print(f"Total samples: {len(labels)}")
    print(f"Image shape: {tactile_images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Number of classes: {len(shape_names)}")
    print(f"\nShape names: {shape_names.tolist()}")
    
    # Check class distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"\n{'='*60}")
    print(f"Class Distribution:")
    print(f"{'='*60}")
    for label, count in zip(unique_labels, counts):
        shape_name = shape_names[label]
        print(f"Class {label:2d} ({shape_name:20s}): {count:3d} samples")
    
    # Check for class imbalance
    min_count = counts.min()
    max_count = counts.max()
    imbalance_ratio = max_count / min_count
    print(f"\nClass balance ratio (max/min): {imbalance_ratio:.2f}")
    if imbalance_ratio > 2.0:
        print("  ⚠️  Warning: Significant class imbalance detected!")
    else:
        print("  ✓ Classes are reasonably balanced")
    
    # Check data validity
    print(f"\n{'='*60}")
    print(f"Data Validity Check:")
    print(f"{'='*60}")
    print(f"Image value range: [{tactile_images.min():.3f}, {tactile_images.max():.3f}]")
    print(f"Image dtype: {tactile_images.dtype}")
    print(f"Labels range: [{labels.min()}, {labels.max()}]")
    print(f"Labels dtype: {labels.dtype}")
    
    # Check for any NaN or inf values
    has_nan = np.any(np.isnan(tactile_images))
    has_inf = np.any(np.isinf(tactile_images))
    print(f"Contains NaN: {has_nan}")
    print(f"Contains Inf: {has_inf}")
    
    if has_nan or has_inf:
        print("  ⚠️  Warning: Dataset contains invalid values!")
        return False
    else:
        print("  ✓ Data is valid")
    
    # Visualize samples if requested
    if visualize:
        print(f"\n{'='*60}")
        print(f"Creating visualization...")
        print(f"{'='*60}")
        
        # Sample one image per class
        fig, axes = plt.subplots(3, 7, figsize=(21, 9))
        axes = axes.flatten()
        
        for i, (label, shape_name) in enumerate(zip(unique_labels, shape_names)):
            # Find first sample of this class
            idx = np.where(labels == label)[0][0]
            img = tactile_images[idx]
            
            # Normalize for display
            if img.max() > 1.0:
                img = img / 255.0
            
            axes[i].imshow(img)
            axes[i].set_title(f"{label}: {shape_name}", fontsize=10)
            axes[i].axis('off')
        
        plt.tight_layout()
        
        # Determine output path
        if output_dir is None:
            output_dir = Path(zarr_path).parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / "dataset_visualization.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
        plt.close()
    
    print(f"\n{'='*60}")
    print(f"Validation complete! Dataset is ready for training.")
    print(f"{'='*60}\n")
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate collected tactile shape dataset"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Path to zarr dataset file'
    )
    parser.add_argument(
        '--no_vis',
        action='store_true',
        help='Disable visualization'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directory to save visualization (default: same as dataset)'
    )
    
    args = parser.parse_args()
    
    validate_dataset(args.dataset, visualize=not args.no_vis, output_dir=args.output_dir)
