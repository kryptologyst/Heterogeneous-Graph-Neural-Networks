"""Test suite for heterogeneous graph neural networks."""

import pytest
import torch
import numpy as np
from torch import Tensor

from src.models import RGCN, HAN, HeterogeneousGNNClassifier
from src.data import HeterogeneousGraphDataset
from src.utils import set_seed, get_device, count_parameters
from src.train import Trainer
from src.eval import ModelEvaluator


class TestModels:
    """Test model implementations."""
    
    def test_rgcn_initialization(self):
        """Test R-GCN model initialization."""
        model = RGCN(
            in_channels=64,
            hidden_channels=32,
            out_channels=16,
            num_relations=5,
            num_layers=2,
        )
        
        assert model.in_channels == 64
        assert model.hidden_channels == 32
        assert model.out_channels == 16
        assert model.num_relations == 5
        assert model.num_layers == 2
        assert len(model.convs) == 2
    
    def test_rgcn_forward(self):
        """Test R-GCN forward pass."""
        set_seed(42)
        
        model = RGCN(
            in_channels=64,
            hidden_channels=32,
            out_channels=16,
            num_relations=5,
            num_layers=2,
        )
        
        # Create dummy data
        num_nodes = 100
        num_edges = 200
        
        x = torch.randn(num_nodes, 64)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_type = torch.randint(0, 5, (num_edges,))
        
        # Forward pass
        output = model(x, edge_index, edge_type)
        
        assert output.shape == (num_nodes, 16)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_han_initialization(self):
        """Test HAN model initialization."""
        model = HAN(
            in_channels=64,
            hidden_channels=32,
            out_channels=16,
            num_node_types=3,
            num_edge_types=5,
            num_layers=2,
            heads=4,
        )
        
        assert model.in_channels == 64
        assert model.hidden_channels == 32
        assert model.out_channels == 16
        assert model.num_node_types == 3
        assert model.num_edge_types == 5
        assert model.num_layers == 2
        assert model.heads == 4
        assert len(model.layers) == 2
    
    def test_han_forward(self):
        """Test HAN forward pass."""
        set_seed(42)
        
        model = HAN(
            in_channels=64,
            hidden_channels=32,
            out_channels=16,
            num_node_types=3,
            num_edge_types=5,
            num_layers=2,
            heads=4,
        )
        
        # Create dummy data
        num_nodes = 100
        num_edges = 200
        
        x = torch.randn(num_nodes, 64)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_type = torch.randint(0, 5, (num_edges,))
        node_type = torch.randint(0, 3, (num_nodes,))
        
        # Forward pass
        output = model(x, edge_index, edge_type, node_type)
        
        assert output.shape == (num_nodes, 16)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_classifier_wrapper(self):
        """Test classifier wrapper."""
        backbone = RGCN(
            in_channels=64,
            hidden_channels=32,
            out_channels=32,
            num_relations=5,
        )
        
        classifier = HeterogeneousGNNClassifier(
            backbone=backbone,
            num_classes=4,
        )
        
        # Create dummy data
        num_nodes = 100
        num_edges = 200
        
        x = torch.randn(num_nodes, 64)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_type = torch.randint(0, 5, (num_edges,))
        
        # Forward pass
        logits = classifier(x, edge_index, edge_type)
        
        assert logits.shape == (num_nodes, 4)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()


class TestData:
    """Test data loading and preprocessing."""
    
    def test_synthetic_dataset(self):
        """Test synthetic dataset generation."""
        dataset = HeterogeneousGraphDataset("synthetic", root="test_data")
        data = dataset.get_data()
        stats = dataset.get_stats()
        
        assert data.num_nodes > 0
        assert data.num_edges > 0
        assert data.num_features > 0
        assert data.num_classes > 0
        assert hasattr(data, 'edge_type')
        assert hasattr(data, 'node_type')
        assert hasattr(data, 'train_mask')
        assert hasattr(data, 'val_mask')
        assert hasattr(data, 'test_mask')
        
        # Check stats
        assert stats["num_nodes"] == data.num_nodes
        assert stats["num_edges"] == data.num_edges
        assert stats["num_features"] == data.num_features
        assert stats["num_classes"] == data.num_classes
    
    def test_data_preprocessing(self):
        """Test data preprocessing."""
        dataset = HeterogeneousGraphDataset("synthetic", root="test_data")
        data = dataset.get_data()
        
        # Test feature normalization
        from src.data import preprocess_data
        
        processed_data = preprocess_data(data, normalize_features=True)
        
        # Check that features are normalized
        feature_mean = processed_data.x.mean(dim=0)
        feature_std = processed_data.x.std(dim=0)
        
        assert torch.allclose(feature_mean, torch.zeros_like(feature_mean), atol=1e-6)
        assert torch.allclose(feature_std, torch.ones_like(feature_std), atol=1e-6)


class TestUtils:
    """Test utility functions."""
    
    def test_seed_setting(self):
        """Test random seed setting."""
        set_seed(42)
        
        # Generate some random numbers
        torch_rand1 = torch.randn(10)
        np_rand1 = np.random.randn(10)
        
        set_seed(42)
        
        # Generate again with same seed
        torch_rand2 = torch.randn(10)
        np_rand2 = np.random.randn(10)
        
        # Should be identical
        assert torch.allclose(torch_rand1, torch_rand2)
        assert np.allclose(np_rand1, np_rand2)
    
    def test_device_selection(self):
        """Test device selection."""
        device = get_device()
        
        assert isinstance(device, torch.device)
        assert device.type in ['cpu', 'cuda', 'mps']
    
    def test_parameter_counting(self):
        """Test parameter counting."""
        model = RGCN(
            in_channels=64,
            hidden_channels=32,
            out_channels=16,
            num_relations=5,
        )
        
        num_params = count_parameters(model)
        assert num_params > 0
        assert isinstance(num_params, int)


class TestTraining:
    """Test training functionality."""
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        model = RGCN(
            in_channels=64,
            hidden_channels=32,
            out_channels=16,
            num_relations=5,
        )
        
        classifier = HeterogeneousGNNClassifier(model, num_classes=4)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)
        
        trainer = Trainer(classifier, optimizer)
        
        assert trainer.model == classifier
        assert trainer.optimizer == optimizer
        assert trainer.device.type in ['cpu', 'cuda', 'mps']
    
    def test_training_step(self):
        """Test single training step."""
        set_seed(42)
        
        # Create model and data
        dataset = HeterogeneousGraphDataset("synthetic", root="test_data")
        data = dataset.get_data()
        
        model = RGCN(
            in_channels=data.num_features,
            hidden_channels=32,
            out_channels=32,
            num_relations=len(data.edge_type.unique()),
        )
        
        classifier = HeterogeneousGNNClassifier(model, num_classes=data.num_classes)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)
        trainer = Trainer(classifier, optimizer)
        
        # Training step
        loss = trainer.train_epoch(data)
        
        assert isinstance(loss, float)
        assert loss >= 0
        assert not np.isnan(loss)
        assert not np.isinf(loss)
    
    def test_validation_step(self):
        """Test validation step."""
        set_seed(42)
        
        # Create model and data
        dataset = HeterogeneousGraphDataset("synthetic", root="test_data")
        data = dataset.get_data()
        
        model = RGCN(
            in_channels=data.num_features,
            hidden_channels=32,
            out_channels=32,
            num_relations=len(data.edge_type.unique()),
        )
        
        classifier = HeterogeneousGNNClassifier(model, num_classes=data.num_classes)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)
        trainer = Trainer(classifier, optimizer)
        
        # Validation step
        val_loss, val_acc = trainer.validate(data)
        
        assert isinstance(val_loss, float)
        assert isinstance(val_acc, float)
        assert val_loss >= 0
        assert 0 <= val_acc <= 1
        assert not np.isnan(val_loss)
        assert not np.isnan(val_acc)


class TestEvaluation:
    """Test evaluation functionality."""
    
    def test_metrics_calculator(self):
        """Test metrics calculator."""
        from src.eval import MetricsCalculator
        
        calc = MetricsCalculator(num_classes=4)
        
        # Create dummy predictions
        y_true = torch.tensor([0, 1, 2, 3, 0, 1])
        y_pred = torch.tensor([0, 1, 2, 3, 0, 1])  # Perfect predictions
        y_prob = torch.softmax(torch.randn(6, 4), dim=1)
        
        metrics = calc.calculate_classification_metrics(y_true, y_pred, y_prob)
        
        assert "accuracy" in metrics
        assert "precision_macro" in metrics
        assert "recall_macro" in metrics
        assert "f1_macro" in metrics
        
        # Perfect predictions should give accuracy = 1.0
        assert metrics["accuracy"] == 1.0
    
    def test_model_evaluator(self):
        """Test model evaluator."""
        set_seed(42)
        
        # Create model and data
        dataset = HeterogeneousGraphDataset("synthetic", root="test_data")
        data = dataset.get_data()
        
        model = RGCN(
            in_channels=data.num_features,
            hidden_channels=32,
            out_channels=32,
            num_relations=len(data.edge_type.unique()),
        )
        
        classifier = HeterogeneousGNNClassifier(model, num_classes=data.num_classes)
        
        evaluator = ModelEvaluator(
            model=classifier,
            device=get_device(),
            num_classes=data.num_classes,
        )
        
        # Evaluate model
        results = evaluator.evaluate_model(data, split="test")
        
        assert "metrics" in results
        metrics = results["metrics"]
        
        assert "accuracy" in metrics
        assert "precision_macro" in metrics
        assert "recall_macro" in metrics
        assert "f1_macro" in metrics
        assert "split" in metrics
        assert metrics["split"] == "test"


if __name__ == "__main__":
    pytest.main([__file__])
