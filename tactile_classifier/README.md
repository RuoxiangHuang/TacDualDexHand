# Tactile Classifier

Self-Attention based CNN classifiers for GelSight tactile sensor classification tasks.

## Overview

This module implements attention-based neural networks for classifying tactile images from GelSight sensors. It supports both single-sensor and multi-sensor configurations with automatic input handling.

## Architecture

### Single-Sensor Classifier (`AttentionPoolingClassifier`)
- **Input**: Tactile RGB images (240×320×3 or custom resolution)
- **Backbone**: 4-layer CNN with BatchNorm and stride-2 convolutions
- **Pooling**: Learnable attention-based spatial pooling
- **Head**: 3-layer MLP with dropout
- **Output**: Class logits

### Multi-Sensor Classifier (`MultiSensorAttentionClassifier`)
- **Input**: Multiple tactile images (N_sensors×3×H×W)
- **Encoder**: Shared CNN backbone for each sensor
- **Fusion**: Multi-head self-attention with positional embeddings
- **Head**: MLP classifier
- **Output**: Class logits with attention weights

### Unified Interface (`TactileAttentionClassifier`)
Automatically selects appropriate architecture based on `num_sensors` parameter.

## Installation

Ensure you have the Isaac Lab environment activated:

```bash
conda activate env_isaaclab
pip install seaborn scikit-learn  # Additional dependencies
```

## Data Collection

Collect tactile data using the shape touching demo:

```bash
# Navigate to project root
cd /path/to/TacDualDexHand

# Run data collection script
./tacex.sh -p scripts/demos/shape_touch/collect_tactile_shape_data.py \
    --auto_mode \
    --samples_per_shape 100 \
    --sample_interval 5 \
    --output_dir ./data/tactile_shapes
```

**Collection Parameters:**
- `--auto_mode`: Enable automatic collection
- `--samples_per_shape`: Number of samples per shape class
- `--sample_interval`: Steps between samples
- `--randomize_pose`: Randomize end-effector poses for diversity

## Data Validation

Validate collected data before training:

```bash
# Activate environment
conda activate env_isaaclab

# Run validation
python data/validate_dataset.py \
    --dataset data/tactile_shapes/YOUR_DATASET.zarr \
    --output_dir data/tactile_shapes/visualizations
```

This checks:
- ✓ Class distribution balance
- ✓ Data integrity (NaN/Inf values)
- ✓ Image statistics
- ✓ Generates visualization samples

## Training

Train the classifier on collected data:

```bash
# Activate environment
conda activate env_isaaclab

# Train with default parameters
python source/train_tactile_classifier.py \
    --dataset data/tactile_shapes/YOUR_DATASET.zarr \
    --epochs 100 \
    --batch_size 16 \
    --lr 1e-3

# Train with custom parameters
python source/train_tactile_classifier.py \
    --dataset data/tactile_shapes/YOUR_DATASET.zarr \
    --epochs 150 \
    --batch_size 32 \
    --lr 5e-4 \
    --dropout 0.4 \
    --val_split 0.25 \
    --output_dir ./models/my_classifier
```

**Training Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset` | Required | Path to zarr dataset |
| `--epochs` | 100 | Number of training epochs |
| `--batch_size` | 16 | Batch size for training |
| `--lr` | 1e-3 | Learning rate |
| `--weight_decay` | 1e-4 | AdamW weight decay |
| `--dropout` | 0.3 | Dropout rate |
| `--val_split` | 0.2 | Validation set ratio |
| `--seed` | 42 | Random seed |
| `--num_workers` | 4 | DataLoader workers |
| `--output_dir` | Auto | Output directory for models |

**Training Outputs:**
- `best_model.pth`: Best model checkpoint (highest val accuracy)
- `final_model_epochN.pth`: Final epoch checkpoint
- `training_curves.png`: Loss and accuracy plots
- `confusion_matrix.png`: Classification confusion matrix
- Console: Classification report with per-class metrics

## Inference

Use the trained classifier in your simulation:

```python
import torch
from tactile_classifier import TactileAttentionClassifier

# Load model
checkpoint = torch.load('path/to/best_model.pth')
model = TactileAttentionClassifier(
    num_classes=checkpoint['num_classes'],
    **checkpoint['model_config']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Inference
tactile_image = ...  # Shape: (B, 3, 240, 320)
pred_class, confidence, probs, attention = model.infer(tactile_image)
```

### Integration Example

See `scripts/demos/shape_touch/run_shape_touch_with_classifier.py` for a complete integration example with visualization.

## Model Architecture Details

### CNN Backbone
```
Input (3×240×320) 
  → Conv(32, k=3, s=2) + BN + ReLU  →  (32×120×160)
  → Conv(64, k=3, s=2) + BN + ReLU  →  (64×60×80)
  → Conv(128, k=3, s=2) + BN + ReLU →  (128×30×40)
  → Conv(256, k=3, s=2) + BN + ReLU →  (256×15×20)
  → Attention Pool                  →  (256)
```

### MLP Head
```
Features (256)
  → Linear(512) + ReLU + Dropout(0.3)
  → Linear(256) + ReLU + Dropout(0.3)
  → Linear(128) + ReLU + Dropout(0.3)
  → Linear(num_classes)
```

**Total Parameters:** ~1.7M (for 21-class classification)

## Performance Tips

1. **Data Collection**: Collect 50-100+ samples per class with pose randomization
2. **Batch Size**: Use largest batch size that fits in memory (16-32 recommended)
3. **Learning Rate**: Start with 1e-3, reduce if training is unstable
4. **Regularization**: Increase dropout (0.4-0.5) if overfitting occurs
5. **Epochs**: Monitor validation accuracy, typically converges in 50-100 epochs

## Citation

If you use this classifier in your research, please cite:

```bibtex
@software{tacdualexhand,
  title={TacDualDexHand: Tactile Sensor Integration for Dual Dexterous Hands},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/TacDualDexHand}
}
```

## License

BSD-3-Clause License
