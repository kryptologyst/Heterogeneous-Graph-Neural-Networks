"""Main training script for heterogeneous graph neural networks."""

import argparse
import os
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

from src.data import HeterogeneousGraphDataset, preprocess_data
from src.models import RGCN, HAN, HeterogeneousGNNClassifier
from src.train import Trainer
from src.eval import ModelEvaluator
from src.utils import set_seed, get_device, get_model_size


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Configuration dictionary.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: Dict[str, Any], data_stats: Dict[str, Any]) -> nn.Module:
    """Create model based on configuration.
    
    Args:
        config: Configuration dictionary.
        data_stats: Dataset statistics.
        
    Returns:
        Model instance.
    """
    model_config = config["model"]
    model_name = model_config["name"].lower()
    
    # Common parameters
    in_channels = data_stats["num_features"]
    hidden_channels = model_config["hidden_channels"]
    out_channels = data_stats["num_classes"]
    num_relations = data_stats["num_edge_types"]
    num_layers = model_config["num_layers"]
    dropout = model_config["dropout"]
    activation = model_config["activation"]
    use_layer_norm = model_config["use_layer_norm"]
    
    if model_name == "rgcn":
        backbone = RGCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            num_relations=num_relations,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            use_bias=model_config["rgcn"]["use_bias"],
            use_self_loops=model_config["rgcn"]["use_self_loops"],
            use_layer_norm=use_layer_norm,
        )
    elif model_name == "han":
        backbone = HAN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            num_node_types=model_config["han"]["num_node_types"],
            num_edge_types=model_config["han"]["num_edge_types"],
            num_layers=num_layers,
            heads=model_config["han"]["heads"],
            dropout=dropout,
            activation=activation,
            use_layer_norm=use_layer_norm,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Wrap with classifier
    model = HeterogeneousGNNClassifier(
        backbone=backbone,
        num_classes=out_channels,
        dropout=dropout,
    )
    
    return model


def create_optimizer(model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
    """Create optimizer based on configuration.
    
    Args:
        model: Model to optimize.
        config: Configuration dictionary.
        
    Returns:
        Optimizer instance.
    """
    training_config = config["training"]
    optimizer_name = training_config["optimizer"].lower()
    lr = training_config["lr"]
    weight_decay = training_config["weight_decay"]
    
    if optimizer_name == "adam":
        optimizer_params = training_config.get("adam", {})
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=optimizer_params.get("betas", [0.9, 0.999]),
            eps=optimizer_params.get("eps", 1e-8),
        )
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.9,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    return optimizer


def create_scheduler(optimizer: optim.Optimizer, config: Dict[str, Any]) -> Any:
    """Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer.
        config: Configuration dictionary.
        
    Returns:
        Scheduler instance or None.
    """
    training_config = config["training"]
    scheduler_name = training_config["scheduler"].lower()
    
    if scheduler_name == "step":
        scheduler_params = training_config.get("scheduler_params", {})
        scheduler = StepLR(
            optimizer,
            step_size=scheduler_params.get("step_size", 30),
            gamma=scheduler_params.get("gamma", 0.1),
        )
    elif scheduler_name == "cosine":
        scheduler_params = training_config.get("scheduler_params", {})
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=scheduler_params.get("T_max", 100),
        )
    else:
        scheduler = None
    
    return scheduler


def create_criterion(config: Dict[str, Any]) -> nn.Module:
    """Create loss function.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        Loss function.
    """
    criterion_name = config["training"]["criterion"].lower()
    
    if criterion_name == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif criterion_name == "nll_loss":
        return nn.NLLLoss()
    else:
        raise ValueError(f"Unknown criterion: {criterion_name}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Heterogeneous GNN")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Dataset name (overrides config)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (overrides config)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs (overrides config)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (overrides config)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (overrides config)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory",
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.data:
        config["data"]["name"] = args.data
    if args.model:
        config["model"]["name"] = args.model
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.lr:
        config["training"]["lr"] = args.lr
    if args.seed:
        config["seed"] = args.seed
    if args.device:
        config["device"]["auto"] = False
        if args.device.lower() == "cpu":
            config["device"]["force_cpu"] = True
        elif args.device.lower() == "cuda":
            config["device"]["force_cuda"] = True
        elif args.device.lower() == "mps":
            config["device"]["force_mps"] = True
    
    # Set random seed
    set_seed(config["seed"])
    
    # Get device
    if config["device"]["auto"]:
        device = get_device()
    elif config["device"]["force_cpu"]:
        device = torch.device("cpu")
    elif config["device"]["force_cuda"]:
        device = torch.device("cuda")
    elif config["device"]["force_mps"]:
        device = torch.device("mps")
    else:
        device = get_device()
    
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    dataset = HeterogeneousGraphDataset(
        name=config["data"]["name"],
        root=config["data"]["root"],
    )
    
    data = dataset.get_data()
    data_stats = dataset.get_stats()
    
    print(f"Dataset statistics: {data_stats}")
    
    # Preprocess data
    data = preprocess_data(
        data,
        normalize_features=config["data"]["normalize_features"],
        add_self_loops=config["data"]["add_self_loops"],
        make_undirected=config["data"]["make_undirected"],
    )
    
    # Create model
    print("Creating model...")
    model = create_model(config, data_stats)
    print(f"Model size: {get_model_size(model)} parameters")
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    criterion = create_criterion(config)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        use_wandb=config["logging"]["use_wandb"],
        project_name=config["logging"]["project_name"],
    )
    
    # Train model
    print("Starting training...")
    checkpoint_path = output_dir / "best_model.pt"
    
    history = trainer.train(
        data=data,
        epochs=config["training"]["epochs"],
        criterion=criterion,
        patience=config["training"]["patience"],
        save_path=str(checkpoint_path),
        verbose=True,
    )
    
    # Update learning rate if scheduler is used
    if scheduler is not None:
        scheduler.step()
    
    # Evaluate model
    print("Evaluating model...")
    evaluator = ModelEvaluator(
        model=model,
        device=device,
        num_classes=data_stats["num_classes"],
    )
    
    # Generate evaluation report
    report = evaluator.generate_report(data)
    print(report)
    
    # Save report
    report_path = output_dir / "evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    
    # Plot results if requested
    if config["evaluation"]["plot_results"]:
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        evaluator.plot_evaluation_results(data, save_dir=str(plots_dir))
    
    # Save final model
    final_model_path = output_dir / "final_model.pt"
    torch.save(model.state_dict(), final_model_path)
    
    print(f"Training completed. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
