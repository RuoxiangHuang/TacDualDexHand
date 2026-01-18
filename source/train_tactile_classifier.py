"""Train tactile shape classifier using collected GelSight data.

This script trains a Self-Attention based CNN classifier to recognize shapes
from tactile images. It supports data augmentation, validation splits, and
automatic model checkpointing.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import zarr
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# Add project root to path for imports
TACEX_PATH = Path(__file__).parent.parent
sys.path.append(str(TACEX_PATH))
from tactile_classifier import TactileAttentionClassifier


class TactileShapeDataset(Dataset):
    """PyTorch dataset for tactile shape classification.
    
    Args:
        images: Tactile images of shape (N, H, W, C).
        labels: Shape class labels of shape (N,).
        transform: Optional transform function.
    """
    
    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        transform: callable | None = None,
    ):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = self.images[idx].astype(np.float32)
        label = self.labels[idx]
        
        # Normalize to [0, 1] if needed
        if img.max() > 1.0:
            img = img / 255.0
        
        # Convert from (H, W, C) to (C, H, W)
        img = np.transpose(img, (2, 0, 1))
        
        # Apply transform if provided
        if self.transform:
            img = self.transform(img)
        
        return torch.from_numpy(img), torch.tensor(label, dtype=torch.long)


def load_zarr_dataset(zarr_path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load tactile dataset from zarr file.
    
    Args:
        zarr_path: Path to zarr dataset directory.
        
    Returns:
        images: Array of shape (N, H, W, C).
        labels: Array of shape (N,).
        shape_names: List of shape class names.
    """
    root = zarr.open(zarr_path, mode='r')
    images = root['tactile_images'][:]
    labels = root['labels'][:]
    shape_names = root['shape_names'][:].tolist()
    
    print(f"Loaded dataset from: {zarr_path}")
    print(f"  Total samples: {len(labels)}")
    print(f"  Image shape: {images.shape}")
    print(f"  Number of classes: {len(shape_names)}")
    
    return images, labels, shape_names


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch.
    
    Args:
        model: Neural network model.
        dataloader: Training data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to run on.
        
    Returns:
        avg_loss: Average training loss.
        accuracy: Training accuracy percentage.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Validate the model.
    
    Args:
        model: Neural network model.
        dataloader: Validation data loader.
        criterion: Loss function.
        device: Device to run on.
        
    Returns:
        avg_loss: Average validation loss.
        accuracy: Validation accuracy percentage.
        predictions: Predicted labels.
        ground_truth: True labels.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)


def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    train_accs: list[float],
    val_accs: list[float],
    save_path: Path,
):
    """Plot and save training curves.
    
    Args:
        train_losses: Training losses per epoch.
        val_losses: Validation losses per epoch.
        train_accs: Training accuracies per epoch.
        val_accs: Validation accuracies per epoch.
        save_path: Path to save the plot.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    ax1.plot(train_losses, label='Train Loss', linewidth=2)
    ax1.plot(val_losses, label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax2.plot(train_accs, label='Train Acc', linewidth=2)
    ax2.plot(val_accs, label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training curves saved to: {save_path}")
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    save_path: Path,
):
    """Plot and save confusion matrix.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        class_names: List of class names.
        save_path: Path to save the plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Confusion matrix saved to: {save_path}")
    plt.close()


def main(args: argparse.Namespace):
    """Main training function.
    
    Args:
        args: Command-line arguments.
    """
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Training Configuration")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print(f"Validation split: {args.val_split}")
    print(f"{'='*60}\n")
    
    # Load data
    images, labels, shape_names = load_zarr_dataset(args.dataset)
    
    # Split data into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels,
        test_size=args.val_split,
        random_state=args.seed,
        stratify=labels
    )
    print(f"\nTrain set: {len(y_train)} samples")
    print(f"Val set: {len(y_val)} samples\n")
    
    # Create datasets and dataloaders
    train_dataset = TactileShapeDataset(X_train, y_train)
    val_dataset = TactileShapeDataset(X_val, y_val)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Create model
    print("Creating model...")
    model = TactileAttentionClassifier(
        num_classes=len(shape_names),
        num_sensors=1,
        image_channels=3,
        hidden_channels=[32, 64, 128, 256],
        mlp_dims=[512, 256, 128],
        dropout=args.dropout
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )
    
    # Training loop
    print(f"{'='*80}")
    print(f"Starting training...")
    print(f"{'='*80}")
    
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate(
            model, val_loader, criterion, device
        )
        
        # Update scheduler
        scheduler.step()
        
        # Record metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Print progress
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch [{epoch+1:3d}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:6.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:6.2f}% | "
            f"LR: {current_lr:.6f}"
        )
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'num_classes': len(shape_names),
                'shape_names': shape_names,
                'model_config': {
                    'num_sensors': 1,
                    'image_channels': 3,
                    'hidden_channels': [32, 64, 128, 256],
                    'mlp_dims': [512, 256, 128],
                    'dropout': args.dropout,
                }
            }
            torch.save(checkpoint, args.output_dir / 'best_model.pth')
            print(f"  → Best model saved (Val Acc: {val_acc:.2f}%)")
    
    print(f"\n{'='*80}")
    print(f"Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"{'='*80}\n")
    
    # Plot training curves
    plot_training_curves(
        train_losses, val_losses, train_accs, val_accs,
        args.output_dir / 'training_curves.png'
    )
    
    # Final evaluation on validation set
    print("Final evaluation on validation set...")
    model.load_state_dict(
        torch.load(args.output_dir / 'best_model.pth')['model_state_dict']
    )
    _, val_acc, val_preds, val_labels = validate(
        model, val_loader, criterion, device
    )
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(
        val_labels, val_preds,
        target_names=shape_names,
        digits=3
    ))
    
    # Plot confusion matrix
    plot_confusion_matrix(
        val_labels, val_preds, shape_names,
        args.output_dir / 'confusion_matrix.png'
    )
    
    # Save final checkpoint
    final_path = args.output_dir / f'final_model_epoch{args.epochs}.pth'
    torch.save(checkpoint, final_path)
    print(f"\nFinal model saved to: {final_path}")
    print(f"Best model saved to: {args.output_dir / 'best_model.pth'}")
    print(f"\nTraining complete! Use best_model.pth for inference.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train tactile shape classifier from collected data"
    )
    
    # Data arguments
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Path to zarr dataset file'
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=None,
        help='Output directory for models and results (default: auto-generated)'
    )
    parser.add_argument(
        '--val_split',
        type=float,
        default=0.2,
        help='Validation split ratio (default: 0.2)'
    )
    
    # Training arguments
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size (default: 16)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Learning rate (default: 1e-3)'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-4,
        help='Weight decay for optimizer (default: 1e-4)'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.3,
        help='Dropout rate (default: 0.3)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of dataloader workers (default: 4)'
    )
    
    args = parser.parse_args()
    
    # Create output directory with timestamp
    if args.output_dir is None:
        dataset_dir = Path(args.dataset).parent
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = dataset_dir / f"training_{timestamp}"
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    main(args)
