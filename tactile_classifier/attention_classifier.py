"""Self-Attention based tactile image classifiers for GelSight sensors.

This module implements various attention-based architectures for classifying
tactile images from GelSight sensors, supporting both single and multi-sensor setups.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union


class AttentionPooling(nn.Module):
    """Attention-based spatial pooling layer.
    
    Replaces global average pooling with learnable attention weights,
    allowing the model to focus on important regions of the tactile image.
    """

    def __init__(self, in_channels: int, hidden_dim: int = 128):
        """Initialize attention pooling layer.
        
        Args:
            in_channels: Number of input feature channels.
            hidden_dim: Hidden dimension for attention computation.
        """
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply attention pooling.
        
        Args:
            x: Input features of shape (B, C, H, W).
            
        Returns:
            pooled: Pooled features of shape (B, C).
            attention_map: Attention weights of shape (B, 1, H, W).
        """
        # Generate attention weights
        attention_map = self.attention(x)  # (B, 1, H, W)
        attention_weights = F.softmax(
            attention_map.flatten(2), dim=2
        ).view_as(attention_map)
        
        # Weighted pooling
        pooled = (x * attention_weights).sum(dim=[2, 3])  # (B, C)
        
        return pooled, attention_weights


class AttentionPoolingClassifier(nn.Module):
    """Single-sensor tactile classifier with attention pooling.
    
    Uses CNN backbone for feature extraction and attention pooling
    for adaptive spatial aggregation, followed by MLP classification head.
    Suitable for single GelSight sensor scenarios.
    
    Supports variable input resolutions. Designed for GelSight Mini sensor
    output resolution (240x320 or 320x240). The CNN backbone uses stride=2
    convolutions and attention pooling, making it resolution-agnostic.
    """

    def __init__(
        self,
        num_classes: int,
        image_channels: int = 3,
        hidden_channels: list[int] = [32, 64, 128, 256],
        mlp_dims: list[int] = [512, 256, 128],
        dropout: float = 0.3,
    ):
        """Initialize attention pooling classifier.
        
        Args:
            num_classes: Number of output classes.
            image_channels: Number of input image channels (default: 3 for RGB).
            hidden_channels: List of channel sizes for CNN backbone.
            mlp_dims: List of dimensions for MLP classification head.
            dropout: Dropout rate for MLP layers.
            
        Note:
            This model accepts variable input resolutions. For GelSight Mini sensors,
            use original resolution (240x320) without resizing.
        """
        super().__init__()
        
        # CNN backbone for feature extraction
        layers = []
        in_c = image_channels
        for out_c in hidden_channels:
            layers.extend([
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
            ])
            in_c = out_c
        self.backbone = nn.Sequential(*layers)
        
        # Attention pooling
        self.attention_pool = AttentionPooling(hidden_channels[-1])
        
        # MLP classification head
        mlp_layers = []
        in_dim = hidden_channels[-1]
        for out_dim in mlp_dims:
            mlp_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = out_dim
        mlp_layers.append(nn.Linear(in_dim, num_classes))
        self.classifier = nn.Sequential(*mlp_layers)

    def forward(
        self, 
        x: torch.Tensor, 
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.
        
        Args:
            x: Input tactile images of shape (B, C, H, W).
                Supports variable resolutions (e.g., 240x320 for GelSight Mini).
            return_attention: Whether to return attention map.
            
        Returns:
            logits: Classification logits of shape (B, num_classes).
            attention_map (optional): Attention weights of shape (B, 1, H', W').
        """
        # Feature extraction
        features = self.backbone(x)  # (B, hidden_channels[-1], H', W')
        
        # Attention pooling
        pooled, attention_map = self.attention_pool(features)  # (B, hidden_channels[-1])
        
        # Classification
        logits = self.classifier(pooled)
        
        if return_attention:
            return logits, attention_map
        return logits

    def infer(
        self, 
        tactile_image: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Inference with probability and attention map.
        
        Args:
            tactile_image: Input tactile image of shape (B, C, H, W).
                Use original sensor resolution (e.g., 240x320) without resizing.
            
        Returns:
            pred_classes: Predicted class indices of shape (B,).
            confidence: Prediction confidence scores of shape (B,).
            probs: Class probabilities of shape (B, num_classes).
            attention_map: Attention weights of shape (B, 1, H', W').
        """
        with torch.no_grad():
            logits, attention_map = self.forward(tactile_image, return_attention=True)
            probs = F.softmax(logits, dim=1)
            pred_classes = torch.argmax(logits, dim=1)
            confidence = probs.max(dim=1)[0]
        return pred_classes, confidence, probs, attention_map


class MultiSensorAttentionClassifier(nn.Module):
    """Multi-sensor tactile classifier with self-attention fusion.
    
    Processes multiple tactile sensors using shared CNN encoders,
    then fuses sensor features via self-attention mechanism to learn
    inter-sensor relationships. Suitable for multi-finger setups like ShadowHand.
    """

    def __init__(
        self,
        num_sensors: int,
        num_classes: int,
        image_size: int = 32,
        image_channels: int = 3,
        sensor_embed_dim: int = 128,
        fusion_dim: int = 256,
        num_attention_heads: int = 4,
        dropout: float = 0.1,
    ):
        """Initialize multi-sensor attention classifier.
        
        Args:
            num_sensors: Number of tactile sensors.
            num_classes: Number of output classes.
            image_size: Input image size (assumes square images).
            image_channels: Number of input image channels.
            sensor_embed_dim: Embedding dimension for each sensor feature.
            fusion_dim: Dimension for fusion MLP.
            num_attention_heads: Number of attention heads.
            dropout: Dropout rate.
        """
        super().__init__()
        
        self.num_sensors = num_sensors
        
        # Shared CNN encoder for each sensor
        self.sensor_encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, sensor_embed_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(sensor_embed_dim),
            nn.AdaptiveAvgPool2d(1),
        )
        
        # Learnable sensor position embeddings
        self.sensor_pos_embed = nn.Parameter(
            torch.randn(1, num_sensors, sensor_embed_dim) * 0.02
        )
        
        # Self-attention layer for inter-sensor fusion
        self.self_attention = nn.MultiheadAttention(
            embed_dim=sensor_embed_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(sensor_embed_dim)
        
        # Fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(sensor_embed_dim * num_sensors, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Classification head
        self.classifier = nn.Linear(fusion_dim // 2, num_classes)

    def forward(
        self,
        tactile_images: torch.Tensor,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.
        
        Args:
            tactile_images: Input tactile images of shape (B, num_sensors, C, H, W).
            return_attention: Whether to return attention weights.
            
        Returns:
            logits: Classification logits of shape (B, num_classes).
            attention_weights (optional): Attention weights of shape (B, num_sensors, num_sensors).
        """
        B = tactile_images.shape[0]
        
        # Encode each sensor
        sensor_features = []
        for i in range(self.num_sensors):
            feat = self.sensor_encoder(tactile_images[:, i])  # (B, embed_dim, 1, 1)
            feat = feat.squeeze(-1).squeeze(-1)  # (B, embed_dim)
            sensor_features.append(feat)
        
        # Stack sensor features: (B, num_sensors, embed_dim)
        sensor_features = torch.stack(sensor_features, dim=1)
        
        # Add position embeddings
        sensor_features = sensor_features + self.sensor_pos_embed
        
        # Self-attention for inter-sensor fusion
        attended_features, attention_weights = self.self_attention(
            sensor_features,  # query
            sensor_features,  # key
            sensor_features,  # value
            need_weights=True,
            average_attn_weights=True,
        )
        
        # Residual connection and layer norm
        sensor_features = self.norm(sensor_features + attended_features)
        
        # Flatten and fuse
        fused = sensor_features.flatten(1)  # (B, num_sensors * embed_dim)
        fused = self.fusion_mlp(fused)  # (B, fusion_dim // 2)
        
        # Classification
        logits = self.classifier(fused)
        
        if return_attention:
            return logits, attention_weights
        return logits

    def infer(
        self,
        tactile_images: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Inference with probability and attention weights.
        
        Args:
            tactile_images: Input tactile images of shape (B, num_sensors, C, H, W).
            
        Returns:
            pred_classes: Predicted class indices of shape (B,).
            confidence: Prediction confidence scores of shape (B,).
            probs: Class probabilities of shape (B, num_classes).
            attention_weights: Inter-sensor attention weights of shape (B, num_sensors, num_sensors).
        """
        with torch.no_grad():
            logits, attention_weights = self.forward(tactile_images, return_attention=True)
            probs = F.softmax(logits, dim=1)
            pred_classes = torch.argmax(logits, dim=1)
            confidence = probs.max(dim=1)[0]
        return pred_classes, confidence, probs, attention_weights


class TactileAttentionClassifier(nn.Module):
    """Unified tactile classifier supporting both single and multi-sensor inputs.
    
    Automatically handles input format and selects appropriate processing path.
    For single sensor: uses attention pooling.
    For multiple sensors: uses self-attention fusion.
    """

    def __init__(
        self,
        num_classes: int,
        num_sensors: int = 1,
        image_size: int = 32,
        image_channels: int = 3,
        **kwargs,
    ):
        """Initialize unified tactile classifier.
        
        Args:
            num_classes: Number of output classes.
            num_sensors: Number of tactile sensors (1 for single, >1 for multi).
            image_size: Input image size (assumes square images).
            image_channels: Number of input image channels.
            **kwargs: Additional arguments passed to underlying models.
        """
        super().__init__()
        
        self.num_sensors = num_sensors
        
        if num_sensors == 1:
            # Single sensor: use attention pooling classifier
            self.model = AttentionPoolingClassifier(
                num_classes=num_classes,
                image_channels=image_channels,
                **kwargs,
            )
        else:
            # Multi-sensor: use self-attention fusion classifier
            self.model = MultiSensorAttentionClassifier(
                num_sensors=num_sensors,
                num_classes=num_classes,
                image_size=image_size,
                image_channels=image_channels,
                **kwargs,
            )

    def forward(
        self,
        tactile_images: torch.Tensor,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with automatic input handling.
        
        Args:
            tactile_images: Input tactile images.
                - Single sensor: (B, C, H, W) or (B, H, W, C)
                - Multi-sensor: (B, num_sensors, C, H, W)
            return_attention: Whether to return attention maps/weights.
            
        Returns:
            logits: Classification logits of shape (B, num_classes).
            attention (optional): Attention maps or weights.
        """
        # Handle input format for single sensor
        if self.num_sensors == 1:
            if len(tactile_images.shape) == 4:
                # Check if (B, H, W, C) format
                if tactile_images.shape[-1] == 3 or tactile_images.shape[1] == 3:
                    if tactile_images.shape[-1] == 3:
                        # (B, H, W, C) -> (B, C, H, W)
                        tactile_images = tactile_images.permute(0, 3, 1, 2)
        
        return self.model(tactile_images, return_attention=return_attention)

    def infer(
        self,
        tactile_images: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Inference with automatic input handling.
        
        Args:
            tactile_images: Input tactile images (format handled automatically).
            
        Returns:
            pred_classes: Predicted class indices of shape (B,).
            confidence: Prediction confidence scores of shape (B,).
            probs: Class probabilities of shape (B, num_classes).
            attention: Attention maps or weights.
        """
        return self.model.infer(tactile_images)
