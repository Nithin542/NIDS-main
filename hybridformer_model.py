"""
HybridFormer: Multi-Modal Neural Network for Network Intrusion Detection

Architecture:
    - CNN Branch: Extracts local patterns from statistical features
    - Transformer Branch: Captures temporal dependencies and sequential patterns  
    - Graph Branch: Models network topology and relationships
    - Fusion Layer: Combines outputs from all branches for final classification

This architecture is designed to leverage different types of features
through specialized neural network branches, achieving better performance
than single-architecture models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import math


class CNNBranch(nn.Module):
    """
    CNN branch for extracting local patterns from statistical features.
    
    Uses 1D convolutions to detect patterns in feature sequences.
    Good for features like packet sizes, byte distributions, etc.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [128, 256, 128],
        kernel_sizes: list = [3, 3, 3],
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.input_dim = input_dim
        
        # Convolutional layers
        layers = []
        in_channels = 1  # Single channel input
        
        for hidden_dim, kernel_size in zip(hidden_dims, kernel_sizes):
            layers.extend([
                nn.Conv1d(
                    in_channels, 
                    hidden_dim, 
                    kernel_size=kernel_size,
                    padding=kernel_size // 2
                ),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_channels = hidden_dim
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Global pooling
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        
        # Output dimension
        self.output_dim = hidden_dims[-1]
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, input_dim) - statistical features
        Returns:
            (batch_size, output_dim) - extracted CNN features
        """
        # Add channel dimension: (batch, 1, input_dim)
        x = x.unsqueeze(1)
        
        # Apply convolutions
        x = self.conv_layers(x)
        
        # Global pooling: (batch, hidden_dim, 1) -> (batch, hidden_dim)
        x = self.global_pool(x).squeeze(-1)
        
        return x


class TransformerBranch(nn.Module):
    """
    Transformer branch for capturing sequential patterns and dependencies.
    
    Uses self-attention to model relationships between different features.
    Good for temporal patterns and feature interactions.
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding (for sequence information)
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output dimension
        self.output_dim = d_model
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, input_dim) - temporal features
        Returns:
            (batch_size, output_dim) - extracted transformer features
        """
        # Project to d_model dimension
        x = self.input_projection(x)  # (batch, d_model)
        
        # Add sequence dimension for transformer
        x = x.unsqueeze(1)  # (batch, 1, d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply transformer
        x = self.transformer_encoder(x)  # (batch, 1, d_model)
        
        # Remove sequence dimension
        x = x.squeeze(1)  # (batch, d_model)
        
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class GraphBranch(nn.Module):
    """
    Graph branch for modeling network topology and relationships.
    
    Uses graph neural network concepts to model connections between features.
    Good for network flow patterns and topology features.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [128, 256, 128],
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.input_dim = input_dim
        
        # Graph convolution layers (simplified as MLPs for feature relationships)
        layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        self.graph_layers = nn.Sequential(*layers)
        
        # Output dimension
        self.output_dim = hidden_dims[-1]
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, input_dim) - graph/topology features
        Returns:
            (batch_size, output_dim) - extracted graph features
        """
        return self.graph_layers(x)


class FusionLayer(nn.Module):
    """
    Fusion layer to combine outputs from multiple branches.
    
    Uses attention-based fusion to weight different branch outputs
    based on their relevance for each sample.
    """
    
    def __init__(
        self,
        branch_dims: Dict[str, int],
        fusion_dim: int = 256,
        num_classes: int = 10,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.branch_names = list(branch_dims.keys())
        total_dim = sum(branch_dims.values())
        
        # Branch-specific attention weights
        self.attention_weights = nn.ModuleDict({
            name: nn.Linear(dim, 1) 
            for name, dim in branch_dims.items()
        })
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.BatchNorm1d(fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Classification head
        self.classifier = nn.Linear(fusion_dim // 2, num_classes)
    
    def forward(self, branch_outputs: Dict[str, torch.Tensor]):
        """
        Args:
            branch_outputs: Dict with keys matching branch_names,
                          values are (batch_size, branch_dim) tensors
        Returns:
            logits: (batch_size, num_classes)
        """
        # Compute attention weights for each branch
        attention_scores = []
        for name in self.branch_names:
            score = self.attention_weights[name](branch_outputs[name])
            attention_scores.append(score)
        
        # Softmax over branches
        attention_scores = torch.cat(attention_scores, dim=1)  # (batch, num_branches)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Apply attention weights
        weighted_outputs = []
        for i, name in enumerate(self.branch_names):
            weight = attention_weights[:, i:i+1]  # (batch, 1)
            weighted = branch_outputs[name] * weight
            weighted_outputs.append(weighted)
        
        # Concatenate all weighted outputs
        fused = torch.cat(weighted_outputs, dim=1)
        
        # Apply fusion layers
        fused = self.fusion(fused)
        
        # Classification
        logits = self.classifier(fused)
        
        return logits


class HybridFormer(nn.Module):
    """
    HybridFormer: Multi-modal neural network combining CNN, Transformer, and Graph branches.
    
    Architecture:
        1. Three specialized branches process different feature subsets
        2. CNN extracts local patterns from statistical features
        3. Transformer captures sequential dependencies
        4. Graph models network topology
        5. Fusion layer combines all outputs with attention
        6. Final classification head
    
    Args:
        cnn_input_dim: Number of features for CNN branch
        transformer_input_dim: Number of features for Transformer branch
        graph_input_dim: Number of features for Graph branch
        num_classes: Number of output classes
        cnn_hidden_dims: Hidden dimensions for CNN layers
        transformer_d_model: Model dimension for Transformer
        transformer_nhead: Number of attention heads
        transformer_layers: Number of transformer layers
        graph_hidden_dims: Hidden dimensions for Graph layers
        fusion_dim: Fusion layer dimension
        dropout: Dropout rate
    
    Example:
        >>> model = HybridFormer(
        ...     cnn_input_dim=20,
        ...     transformer_input_dim=20,
        ...     graph_input_dim=15,
        ...     num_classes=10
        ... )
        >>> cnn_features = torch.randn(32, 20)
        >>> transformer_features = torch.randn(32, 20)
        >>> graph_features = torch.randn(32, 15)
        >>> logits = model(cnn_features, transformer_features, graph_features)
        >>> logits.shape
        torch.Size([32, 10])
    """
    
    def __init__(
        self,
        cnn_input_dim: int,
        transformer_input_dim: int,
        graph_input_dim: int,
        num_classes: int = 10,
        cnn_hidden_dims: list = [128, 256, 128],
        cnn_kernel_sizes: list = [3, 3, 3],
        transformer_d_model: int = 128,
        transformer_nhead: int = 4,
        transformer_layers: int = 2,
        transformer_dim_feedforward: int = 256,
        graph_hidden_dims: list = [128, 256, 128],
        fusion_dim: int = 256,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Store configuration
        self.cnn_input_dim = cnn_input_dim
        self.transformer_input_dim = transformer_input_dim
        self.graph_input_dim = graph_input_dim
        self.num_classes = num_classes
        
        # CNN Branch
        self.cnn_branch = CNNBranch(
            input_dim=cnn_input_dim,
            hidden_dims=cnn_hidden_dims,
            kernel_sizes=cnn_kernel_sizes,
            dropout=dropout
        )
        
        # Transformer Branch
        self.transformer_branch = TransformerBranch(
            input_dim=transformer_input_dim,
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            num_layers=transformer_layers,
            dim_feedforward=transformer_dim_feedforward,
            dropout=dropout
        )
        
        # Graph Branch
        self.graph_branch = GraphBranch(
            input_dim=graph_input_dim,
            hidden_dims=graph_hidden_dims,
            dropout=dropout
        )
        
        # Fusion Layer
        branch_dims = {
            'cnn': self.cnn_branch.output_dim,
            'transformer': self.transformer_branch.output_dim,
            'graph': self.graph_branch.output_dim
        }
        self.fusion = FusionLayer(
            branch_dims=branch_dims,
            fusion_dim=fusion_dim,
            num_classes=num_classes,
            dropout=dropout
        )
    
    def forward(
        self,
        cnn_features: torch.Tensor,
        transformer_features: torch.Tensor,
        graph_features: torch.Tensor
    ):
        """
        Forward pass through HybridFormer.
        
        Args:
            cnn_features: (batch_size, cnn_input_dim)
            transformer_features: (batch_size, transformer_input_dim)
            graph_features: (batch_size, graph_input_dim)
        
        Returns:
            logits: (batch_size, num_classes)
        """
        # Process each branch
        cnn_output = self.cnn_branch(cnn_features)
        transformer_output = self.transformer_branch(transformer_features)
        graph_output = self.graph_branch(graph_features)
        
        # Fuse outputs
        branch_outputs = {
            'cnn': cnn_output,
            'transformer': transformer_output,
            'graph': graph_output
        }
        
        logits = self.fusion(branch_outputs)
        
        return logits
    
    def get_branch_outputs(
        self,
        cnn_features: torch.Tensor,
        transformer_features: torch.Tensor,
        graph_features: torch.Tensor
    ):
        """
        Get intermediate branch outputs for analysis/visualization.
        
        Returns:
            Dict with branch outputs and final logits
        """
        cnn_output = self.cnn_branch(cnn_features)
        transformer_output = self.transformer_branch(transformer_features)
        graph_output = self.graph_branch(graph_features)
        
        branch_outputs = {
            'cnn': cnn_output,
            'transformer': transformer_output,
            'graph': graph_output
        }
        
        logits = self.fusion(branch_outputs)
        
        return {
            **branch_outputs,
            'logits': logits
        }
