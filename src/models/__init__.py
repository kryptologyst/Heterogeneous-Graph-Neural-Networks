"""Heterogeneous Graph Neural Network models."""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import RGCNConv, HeteroConv, GCNConv, GATConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor


class RGCN(nn.Module):
    """Relational Graph Convolutional Network (R-GCN).
    
    A baseline model for heterogeneous graphs that handles different edge types
    through separate weight matrices for each relation.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_relations: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        activation: str = "relu",
        use_bias: bool = True,
        use_self_loops: bool = True,
        use_layer_norm: bool = False,
    ):
        """Initialize R-GCN model.
        
        Args:
            in_channels: Input feature dimension.
            hidden_channels: Hidden feature dimension.
            out_channels: Output feature dimension.
            num_relations: Number of relation types.
            num_layers: Number of GCN layers.
            dropout: Dropout rate.
            activation: Activation function.
            use_bias: Whether to use bias.
            use_self_loops: Whether to use self-loops.
            use_layer_norm: Whether to use layer normalization.
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_layer_norm = use_layer_norm
        
        # Activation function
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        elif activation == "tanh":
            self.activation = torch.tanh
        else:
            self.activation = lambda x: x
        
        # Build layers
        self.convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList() if use_layer_norm else None
        
        # Input layer
        self.convs.append(
            RGCNConv(
                in_channels,
                hidden_channels,
                num_relations,
                bias=use_bias,
                add_self_loops=use_self_loops,
            )
        )
        
        if use_layer_norm:
            self.layer_norms.append(nn.LayerNorm(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                RGCNConv(
                    hidden_channels,
                    hidden_channels,
                    num_relations,
                    bias=use_bias,
                    add_self_loops=use_self_loops,
                )
            )
            if use_layer_norm:
                self.layer_norms.append(nn.LayerNorm(hidden_channels))
        
        # Output layer
        if num_layers > 1:
            self.convs.append(
                RGCNConv(
                    hidden_channels,
                    out_channels,
                    num_relations,
                    bias=use_bias,
                    add_self_loops=use_self_loops,
                )
            )
    
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
    ) -> Tensor:
        """Forward pass.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            edge_type: Edge types.
            
        Returns:
            Node embeddings.
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)
            
            if i < len(self.convs) - 1:  # Don't apply activation to last layer
                if self.layer_norms is not None:
                    x = self.layer_norms[i](x)
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x


class HANLayer(nn.Module):
    """Heterogeneous Attention Network Layer.
    
    Implements attention mechanism for heterogeneous graphs with different
    node and edge types.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_node_types: int,
        num_edge_types: int,
        heads: int = 1,
        dropout: float = 0.0,
        use_bias: bool = True,
    ):
        """Initialize HAN layer.
        
        Args:
            in_channels: Input feature dimension.
            out_channels: Output feature dimension.
            num_node_types: Number of node types.
            num_edge_types: Number of edge types.
            heads: Number of attention heads.
            dropout: Dropout rate.
            use_bias: Whether to use bias.
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.heads = heads
        self.dropout = dropout
        
        # Node type-specific transformations
        self.node_transforms = nn.ModuleDict()
        for node_type in range(num_node_types):
            self.node_transforms[f"type_{node_type}"] = Linear(
                in_channels, out_channels, bias=use_bias
            )
        
        # Edge type-specific attention weights
        self.attention_weights = nn.Parameter(
            torch.randn(num_edge_types, heads, out_channels, out_channels)
        )
        
        # Output projection
        self.out_proj = Linear(out_channels * heads, out_channels, bias=use_bias)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Reset parameters."""
        for transform in self.node_transforms.values():
            transform.reset_parameters()
        nn.init.xavier_uniform_(self.attention_weights)
        self.out_proj.reset_parameters()
    
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        node_type: Tensor,
    ) -> Tensor:
        """Forward pass.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            edge_type: Edge types.
            node_type: Node types.
            
        Returns:
            Updated node embeddings.
        """
        # Transform features by node type
        x_transformed = torch.zeros(
            x.size(0), self.out_channels, device=x.device, dtype=x.dtype
        )
        
        for node_type_idx in range(self.num_node_types):
            mask = node_type == node_type_idx
            if mask.any():
                x_transformed[mask] = self.node_transforms[f"type_{node_type_idx}"](x[mask])
        
        # Compute attention weights
        src, dst = edge_index[0], edge_index[1]
        attention_scores = torch.zeros(
            edge_index.size(1), self.heads, device=x.device, dtype=x.dtype
        )
        
        for edge_type_idx in range(self.num_edge_types):
            mask = edge_type == edge_type_idx
            if mask.any():
                edge_attention = self.attention_weights[edge_type_idx]
                for head in range(self.heads):
                    scores = torch.sum(
                        x_transformed[src[mask]] * edge_attention[head] * x_transformed[dst[mask]],
                        dim=1
                    )
                    attention_scores[mask, head] = scores
        
        # Apply softmax
        attention_weights = F.softmax(attention_scores, dim=0)
        
        # Aggregate messages
        out = torch.zeros(
            x.size(0), self.out_channels * self.heads, device=x.device, dtype=x.dtype
        )
        
        for head in range(self.heads):
            head_out = torch.zeros(
                x.size(0), self.out_channels, device=x.device, dtype=x.dtype
            )
            
            for edge_type_idx in range(self.num_edge_types):
                mask = edge_type == edge_type_idx
                if mask.any():
                    src_nodes = src[mask]
                    dst_nodes = dst[mask]
                    weights = attention_weights[mask, head:head+1]
                    
                    # Aggregate messages
                    head_out.scatter_add_(
                        0,
                        dst_nodes.unsqueeze(1).expand(-1, self.out_channels),
                        weights * x_transformed[src_nodes]
                    )
            
            out[:, head * self.out_channels:(head + 1) * self.out_channels] = head_out
        
        # Apply dropout and output projection
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.out_proj(out)
        
        return out


class HAN(nn.Module):
    """Heterogeneous Attention Network.
    
    A more sophisticated model for heterogeneous graphs that uses attention
    mechanisms to handle different node and edge types.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_node_types: int,
        num_edge_types: int,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.0,
        activation: str = "relu",
        use_layer_norm: bool = True,
    ):
        """Initialize HAN model.
        
        Args:
            in_channels: Input feature dimension.
            hidden_channels: Hidden feature dimension.
            out_channels: Output feature dimension.
            num_node_types: Number of node types.
            num_edge_types: Number of edge types.
            num_layers: Number of layers.
            heads: Number of attention heads.
            dropout: Dropout rate.
            activation: Activation function.
            use_layer_norm: Whether to use layer normalization.
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        self.use_layer_norm = use_layer_norm
        
        # Activation function
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        elif activation == "tanh":
            self.activation = torch.tanh
        else:
            self.activation = lambda x: x
        
        # Build layers
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList() if use_layer_norm else None
        
        # Input layer
        self.layers.append(
            HANLayer(
                in_channels,
                hidden_channels,
                num_node_types,
                num_edge_types,
                heads,
                dropout,
            )
        )
        
        if use_layer_norm:
            self.layer_norms.append(nn.LayerNorm(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(
                HANLayer(
                    hidden_channels,
                    hidden_channels,
                    num_node_types,
                    num_edge_types,
                    heads,
                    dropout,
                )
            )
            if use_layer_norm:
                self.layer_norms.append(nn.LayerNorm(hidden_channels))
        
        # Output layer
        if num_layers > 1:
            self.layers.append(
                HANLayer(
                    hidden_channels,
                    out_channels,
                    num_node_types,
                    num_edge_types,
                    heads,
                    dropout,
                )
            )
    
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        node_type: Tensor,
    ) -> Tensor:
        """Forward pass.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            edge_type: Edge types.
            node_type: Node types.
            
        Returns:
            Node embeddings.
        """
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_type, node_type)
            
            if i < len(self.layers) - 1:  # Don't apply activation to last layer
                if self.layer_norms is not None:
                    x = self.layer_norms[i](x)
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x


class HeterogeneousGNNClassifier(nn.Module):
    """Heterogeneous GNN classifier wrapper."""
    
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        dropout: float = 0.5,
    ):
        """Initialize classifier.
        
        Args:
            backbone: Backbone GNN model.
            num_classes: Number of classes.
            dropout: Dropout rate.
        """
        super().__init__()
        
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(backbone.out_channels, num_classes),
        )
    
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        node_type: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            edge_type: Edge types.
            node_type: Optional node types.
            
        Returns:
            Class logits.
        """
        # Get node embeddings
        if isinstance(self.backbone, HAN):
            embeddings = self.backbone(x, edge_index, edge_type, node_type)
        else:  # RGCN
            embeddings = self.backbone(x, edge_index, edge_type)
        
        # Classify
        logits = self.classifier(embeddings)
        
        return logits
