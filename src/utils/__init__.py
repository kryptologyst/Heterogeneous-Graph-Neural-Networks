"""Utility functions for heterogeneous graph neural networks."""

import random
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device with fallback chain.
    
    Returns:
        Available device (CUDA -> MPS -> CPU).
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model.
        
    Returns:
        Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size(model: nn.Module) -> str:
    """Get human-readable model size.
    
    Args:
        model: PyTorch model.
        
    Returns:
        Model size as string (e.g., "1.2M").
    """
    num_params = count_parameters(model)
    
    if num_params >= 1e6:
        return f"{num_params / 1e6:.1f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.1f}K"
    else:
        return str(num_params)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    metrics: Dict[str, float],
    filepath: str,
) -> None:
    """Save model checkpoint.
    
    Args:
        model: PyTorch model.
        optimizer: Optimizer.
        epoch: Current epoch.
        loss: Current loss.
        metrics: Dictionary of metrics.
        filepath: Path to save checkpoint.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "metrics": metrics,
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[int, float, Dict[str, float]]:
    """Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint file.
        model: PyTorch model.
        optimizer: Optional optimizer.
        
    Returns:
        Tuple of (epoch, loss, metrics).
    """
    checkpoint = torch.load(filepath, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return checkpoint["epoch"], checkpoint["loss"], checkpoint["metrics"]


def normalize_features(x: Tensor) -> Tensor:
    """Normalize node features.
    
    Args:
        x: Node feature matrix.
        
    Returns:
        Normalized features.
    """
    return (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-8)


def create_edge_type_mapping(edge_types: Tensor) -> Dict[int, str]:
    """Create mapping from edge type indices to names.
    
    Args:
        edge_types: Edge type tensor.
        
    Returns:
        Mapping from index to edge type name.
    """
    unique_types = edge_types.unique().cpu().numpy()
    return {i: f"relation_{i}" for i in unique_types}


def create_node_type_mapping(node_types: Optional[Tensor] = None) -> Dict[int, str]:
    """Create mapping from node type indices to names.
    
    Args:
        node_types: Optional node type tensor.
        
    Returns:
        Mapping from index to node type name.
    """
    if node_types is None:
        return {0: "entity"}
    
    unique_types = node_types.unique().cpu().numpy()
    return {i: f"node_type_{i}" for i in unique_types}
