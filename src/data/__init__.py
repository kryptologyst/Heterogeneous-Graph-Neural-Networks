"""Data loading and preprocessing utilities for heterogeneous graphs."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch_geometric.data import Data, HeteroData
from torch_geometric.datasets import AIFBDataset, AMDataset, MUTAGDataset
from torch_geometric.loader import DataLoader, NeighborLoader
from torch_geometric.transforms import NormalizeFeatures, ToUndirected
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class HeterogeneousGraphDataset:
    """Dataset wrapper for heterogeneous graph data."""
    
    def __init__(
        self,
        name: str = "aifb",
        root: str = "data",
        transform: Optional[Any] = None,
        pre_transform: Optional[Any] = None,
    ):
        """Initialize dataset.
        
        Args:
            name: Dataset name ('aifb', 'am', 'mutag', or 'synthetic').
            root: Root directory for data.
            transform: Optional transform to apply.
            pre_transform: Optional pre-transform to apply.
        """
        self.name = name
        self.root = Path(root)
        self.transform = transform
        self.pre_transform = pre_transform
        
        # Create data directory
        self.root.mkdir(parents=True, exist_ok=True)
        
        self._load_dataset()
    
    def _load_dataset(self) -> None:
        """Load the specified dataset."""
        if self.name == "aifb":
            self._load_aifb()
        elif self.name == "am":
            self._load_am()
        elif self.name == "mutag":
            self._load_mutag()
        elif self.name == "synthetic":
            self._generate_synthetic()
        else:
            raise ValueError(f"Unknown dataset: {self.name}")
    
    def _load_aifb(self) -> None:
        """Load AIFB dataset."""
        dataset = AIFBDataset(root=str(self.root / "aifb"))
        self.data = dataset[0]
        
        # Encode labels
        if hasattr(self.data, 'y') and self.data.y is not None:
            le = LabelEncoder()
            self.data.y = torch.from_numpy(le.fit_transform(self.data.y.numpy()))
        
        # Add synthetic node types if not present
        if not hasattr(self.data, 'node_type'):
            self.data.node_type = torch.zeros(self.data.num_nodes, dtype=torch.long)
    
    def _load_am(self) -> None:
        """Load AM dataset."""
        dataset = AMDataset(root=str(self.root / "am"))
        self.data = dataset[0]
        
        # Encode labels
        if hasattr(self.data, 'y') and self.data.y is not None:
            le = LabelEncoder()
            self.data.y = torch.from_numpy(le.fit_transform(self.data.y.numpy()))
        
        # Add synthetic node types if not present
        if not hasattr(self.data, 'node_type'):
            self.data.node_type = torch.zeros(self.data.num_nodes, dtype=torch.long)
    
    def _load_mutag(self) -> None:
        """Load MUTAG dataset."""
        dataset = MUTAGDataset(root=str(self.root / "mutag"))
        self.data = dataset[0]
        
        # Encode labels
        if hasattr(self.data, 'y') and self.data.y is not None:
            le = LabelEncoder()
            self.data.y = torch.from_numpy(le.fit_transform(self.data.y.numpy()))
        
        # Add synthetic node types if not present
        if not hasattr(self.data, 'node_type'):
            self.data.node_type = torch.zeros(self.data.num_nodes, dtype=torch.long)
    
    def _generate_synthetic(self) -> None:
        """Generate synthetic heterogeneous graph."""
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Parameters
        num_nodes = 1000
        num_node_types = 3
        num_edge_types = 5
        num_classes = 4
        feature_dim = 64
        
        # Generate node features and types
        x = torch.randn(num_nodes, feature_dim)
        node_type = torch.randint(0, num_node_types, (num_nodes,))
        
        # Generate edges with different types
        edge_indices = []
        edge_types = []
        
        for edge_type in range(num_edge_types):
            # Different connectivity patterns for different edge types
            if edge_type == 0:  # Dense connections
                num_edges = num_nodes * 2
            elif edge_type == 1:  # Sparse connections
                num_edges = num_nodes // 2
            else:  # Medium connections
                num_edges = num_nodes
            
            src = torch.randint(0, num_nodes, (num_edges,))
            dst = torch.randint(0, num_nodes, (num_edges,))
            
            edge_indices.append(torch.stack([src, dst]))
            edge_types.append(torch.full((num_edges,), edge_type, dtype=torch.long))
        
        edge_index = torch.cat(edge_indices, dim=1)
        edge_type = torch.cat(edge_types)
        
        # Generate labels (node classification)
        y = torch.randint(0, num_classes, (num_nodes,))
        
        # Create train/val/test splits
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        indices = torch.randperm(num_nodes)
        train_size = int(0.6 * num_nodes)
        val_size = int(0.2 * num_nodes)
        
        train_mask[indices[:train_size]] = True
        val_mask[indices[train_size:train_size + val_size]] = True
        test_mask[indices[train_size + val_size:]] = True
        
        self.data = Data(
            x=x,
            edge_index=edge_index,
            edge_type=edge_type,
            node_type=node_type,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )
    
    def get_data(self) -> Data:
        """Get the loaded data.
        
        Returns:
            PyTorch Geometric Data object.
        """
        return self.data
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics.
        
        Returns:
            Dictionary of dataset statistics.
        """
        data = self.data
        
        stats = {
            "num_nodes": data.num_nodes,
            "num_edges": data.num_edges,
            "num_features": data.num_features,
            "num_classes": len(torch.unique(data.y)) if hasattr(data, 'y') else 0,
            "num_edge_types": len(torch.unique(data.edge_type)) if hasattr(data, 'edge_type') else 0,
            "num_node_types": len(torch.unique(data.node_type)) if hasattr(data, 'node_type') else 0,
        }
        
        if hasattr(data, 'train_mask'):
            stats.update({
                "train_nodes": data.train_mask.sum().item(),
                "val_nodes": data.val_mask.sum().item(),
                "test_nodes": data.test_mask.sum().item(),
            })
        
        return stats


def create_data_loaders(
    data: Data,
    batch_size: int = 32,
    num_workers: int = 0,
    neighbor_sampling: bool = False,
    fanouts: Optional[List[int]] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders for training, validation, and testing.
    
    Args:
        data: Graph data.
        batch_size: Batch size.
        num_workers: Number of worker processes.
        neighbor_sampling: Whether to use neighbor sampling.
        fanouts: Fanout values for neighbor sampling.
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    if neighbor_sampling and fanouts is not None:
        # Use neighbor sampling for large graphs
        train_loader = NeighborLoader(
            data,
            num_neighbors=fanouts,
            batch_size=batch_size,
            input_nodes=data.train_mask,
            shuffle=True,
            num_workers=num_workers,
        )
        
        val_loader = NeighborLoader(
            data,
            num_neighbors=fanouts,
            batch_size=batch_size,
            input_nodes=data.val_mask,
            shuffle=False,
            num_workers=num_workers,
        )
        
        test_loader = NeighborLoader(
            data,
            num_neighbors=fanouts,
            batch_size=batch_size,
            input_nodes=data.test_mask,
            shuffle=False,
            num_workers=num_workers,
        )
    else:
        # Use standard data loading
        train_loader = DataLoader([data], batch_size=1, shuffle=False)
        val_loader = DataLoader([data], batch_size=1, shuffle=False)
        test_loader = DataLoader([data], batch_size=1, shuffle=False)
    
    return train_loader, val_loader, test_loader


def preprocess_data(
    data: Data,
    normalize_features: bool = True,
    add_self_loops: bool = True,
    make_undirected: bool = False,
) -> Data:
    """Preprocess graph data.
    
    Args:
        data: Input graph data.
        normalize_features: Whether to normalize node features.
        add_self_loops: Whether to add self-loops.
        make_undirected: Whether to make graph undirected.
        
    Returns:
        Preprocessed data.
    """
    if normalize_features:
        data.x = (data.x - data.x.mean(dim=0)) / (data.x.std(dim=0) + 1e-8)
    
    if make_undirected:
        transform = ToUndirected()
        data = transform(data)
    
    return data
