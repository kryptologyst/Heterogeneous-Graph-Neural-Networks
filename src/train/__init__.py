"""Training utilities for heterogeneous graph neural networks."""

import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import get_device, save_checkpoint, load_checkpoint


class Trainer:
    """Trainer class for heterogeneous GNN models."""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        device: Optional[torch.device] = None,
        use_wandb: bool = False,
        project_name: str = "heterogeneous-gnn",
    ):
        """Initialize trainer.
        
        Args:
            model: Model to train.
            optimizer: Optimizer.
            device: Device to use.
            use_wandb: Whether to use Weights & Biases logging.
            project_name: W&B project name.
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device or get_device()
        self.use_wandb = use_wandb
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize W&B if requested
        if use_wandb:
            try:
                import wandb
                wandb.init(project=project_name, config={})
                self.wandb = wandb
            except ImportError:
                print("Warning: wandb not available, logging disabled")
                self.use_wandb = False
                self.wandb = None
        else:
            self.wandb = None
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_val_acc = 0.0
        self.best_epoch = 0
    
    def train_epoch(
        self,
        data: Any,
        criterion: nn.Module = nn.CrossEntropyLoss(),
    ) -> float:
        """Train for one epoch.
        
        Args:
            data: Graph data.
            criterion: Loss function.
            
        Returns:
            Average training loss.
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move data to device
        data = data.to(self.device)
        
        # Forward pass
        if hasattr(data, 'node_type'):
            logits = self.model(data.x, data.edge_index, data.edge_type, data.node_type)
        else:
            logits = self.model(data.x, data.edge_index, data.edge_type)
        
        # Compute loss
        loss = criterion(logits[data.train_mask], data.y[data.train_mask])
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def validate(
        self,
        data: Any,
        criterion: nn.Module = nn.CrossEntropyLoss(),
    ) -> Tuple[float, float]:
        """Validate the model.
        
        Args:
            data: Graph data.
            criterion: Loss function.
            
        Returns:
            Tuple of (validation loss, validation accuracy).
        """
        self.model.eval()
        
        with torch.no_grad():
            # Move data to device
            data = data.to(self.device)
            
            # Forward pass
            if hasattr(data, 'node_type'):
                logits = self.model(data.x, data.edge_index, data.edge_type, data.node_type)
            else:
                logits = self.model(data.x, data.edge_index, data.edge_type)
            
            # Compute loss
            val_loss = criterion(logits[data.val_mask], data.y[data.val_mask])
            
            # Compute accuracy
            pred = logits[data.val_mask].argmax(dim=1)
            correct = pred == data.y[data.val_mask]
            val_acc = correct.float().mean().item()
        
        return val_loss.item(), val_acc
    
    def test(
        self,
        data: Any,
        criterion: nn.Module = nn.CrossEntropyLoss(),
    ) -> Tuple[float, float]:
        """Test the model.
        
        Args:
            data: Graph data.
            criterion: Loss function.
            
        Returns:
            Tuple of (test loss, test accuracy).
        """
        self.model.eval()
        
        with torch.no_grad():
            # Move data to device
            data = data.to(self.device)
            
            # Forward pass
            if hasattr(data, 'node_type'):
                logits = self.model(data.x, data.edge_index, data.edge_type, data.node_type)
            else:
                logits = self.model(data.x, data.edge_index, data.edge_type)
            
            # Compute loss
            test_loss = criterion(logits[data.test_mask], data.y[data.test_mask])
            
            # Compute accuracy
            pred = logits[data.test_mask].argmax(dim=1)
            correct = pred == data.y[data.test_mask]
            test_acc = correct.float().mean().item()
        
        return test_loss, test_acc
    
    def train(
        self,
        data: Any,
        epochs: int = 100,
        criterion: nn.Module = nn.CrossEntropyLoss(),
        patience: int = 10,
        save_path: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Train the model.
        
        Args:
            data: Graph data.
            epochs: Number of epochs.
            criterion: Loss function.
            patience: Early stopping patience.
            save_path: Path to save best model.
            verbose: Whether to print progress.
            
        Returns:
            Training history.
        """
        if verbose:
            print(f"Training on {self.device}")
            print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Training loop
        start_time = time.time()
        patience_counter = 0
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Train
            train_loss = self.train_epoch(data, criterion)
            
            # Validate
            val_loss, val_acc = self.validate(data, criterion)
            
            # Update history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Check for best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                patience_counter = 0
                
                # Save best model
                if save_path:
                    save_checkpoint(
                        self.model,
                        self.optimizer,
                        epoch,
                        val_loss,
                        {"val_acc": val_acc},
                        save_path,
                    )
            else:
                patience_counter += 1
            
            # Log metrics
            epoch_time = time.time() - epoch_start
            
            if self.use_wandb and self.wandb:
                self.wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "epoch_time": epoch_time,
                })
            
            if verbose and (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1:3d}/{epochs}: "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Val Acc: {val_acc:.4f}, "
                    f"Time: {epoch_time:.2f}s"
                )
            
            # Early stopping
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
        
        total_time = time.time() - start_time
        
        if verbose:
            print(f"\nTraining completed in {total_time:.2f}s")
            print(f"Best validation accuracy: {self.best_val_acc:.4f} at epoch {self.best_epoch+1}")
        
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_accuracies": self.val_accuracies,
            "best_val_acc": self.best_val_acc,
            "best_epoch": self.best_epoch,
            "total_time": total_time,
        }
    
    def evaluate(
        self,
        data: Any,
        criterion: nn.Module = nn.CrossEntropyLoss(),
        load_best: bool = True,
        checkpoint_path: Optional[str] = None,
    ) -> Dict[str, float]:
        """Evaluate the model.
        
        Args:
            data: Graph data.
            criterion: Loss function.
            load_best: Whether to load best checkpoint.
            checkpoint_path: Path to checkpoint.
            
        Returns:
            Evaluation metrics.
        """
        if load_best and checkpoint_path:
            load_checkpoint(checkpoint_path, self.model, self.optimizer)
        
        # Test
        test_loss, test_acc = self.test(data, criterion)
        
        # Validation
        val_loss, val_acc = self.validate(data, criterion)
        
        metrics = {
            "test_loss": test_loss,
            "test_acc": test_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        
        return metrics
