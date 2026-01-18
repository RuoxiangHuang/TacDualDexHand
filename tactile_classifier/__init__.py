"""Tactile classifier module for GelSight sensors.

This module provides neural network models for tactile image classification
using self-attention mechanisms.
"""

from .attention_classifier import (
    TactileAttentionClassifier,
    MultiSensorAttentionClassifier,
    AttentionPoolingClassifier,
)

__all__ = [
    "TactileAttentionClassifier",
    "MultiSensorAttentionClassifier",
    "AttentionPoolingClassifier",
]
