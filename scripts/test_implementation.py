#!/usr/bin/env python3
"""Quick test script to verify the implementation works."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import HeterogeneousGraphDataset
from src.models import RGCN, HAN, HeterogeneousGNNClassifier
from src.train import Trainer
from src.eval import ModelEvaluator
from src.utils import set_seed, get_device, get_model_size


def test_implementation():
    """Test the complete implementation."""
    print("🧪 Testing Heterogeneous GNN Implementation")
    print("=" * 50)
    
    # Set seed for reproducibility
    set_seed(42)
    device = get_device()
    print(f"📱 Using device: {device}")
    
    # Test data loading
    print("\n📊 Testing data loading...")
    try:
        dataset = HeterogeneousGraphDataset("synthetic", root="test_data")
        data = dataset.get_data()
        stats = dataset.get_stats()
        
        print(f"✅ Dataset loaded successfully!")
        print(f"   - Nodes: {stats['num_nodes']}")
        print(f"   - Edges: {stats['num_edges']}")
        print(f"   - Features: {stats['num_features']}")
        print(f"   - Classes: {stats['num_classes']}")
        print(f"   - Edge Types: {stats['num_edge_types']}")
        print(f"   - Node Types: {stats['num_node_types']}")
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False
    
    # Test R-GCN model
    print("\n🔧 Testing R-GCN model...")
    try:
        rgcn_backbone = RGCN(
            in_channels=stats["num_features"],
            hidden_channels=32,
            out_channels=32,
            num_relations=stats["num_edge_types"],
            num_layers=2,
        )
        
        rgcn_model = HeterogeneousGNNClassifier(
            backbone=rgcn_backbone,
            num_classes=stats["num_classes"],
        )
        
        print(f"✅ R-GCN model created successfully!")
        print(f"   - Parameters: {get_model_size(rgcn_model)}")
        
        # Test forward pass
        rgcn_model.eval()
        with torch.no_grad():
            logits = rgcn_model(data.x, data.edge_index, data.edge_type)
            print(f"   - Output shape: {logits.shape}")
            
    except Exception as e:
        print(f"❌ R-GCN model failed: {e}")
        return False
    
    # Test HAN model
    print("\n🔧 Testing HAN model...")
    try:
        han_backbone = HAN(
            in_channels=stats["num_features"],
            hidden_channels=32,
            out_channels=32,
            num_node_types=stats["num_node_types"],
            num_edge_types=stats["num_edge_types"],
            num_layers=2,
            heads=4,
        )
        
        han_model = HeterogeneousGNNClassifier(
            backbone=han_backbone,
            num_classes=stats["num_classes"],
        )
        
        print(f"✅ HAN model created successfully!")
        print(f"   - Parameters: {get_model_size(han_model)}")
        
        # Test forward pass
        han_model.eval()
        with torch.no_grad():
            logits = han_model(data.x, data.edge_index, data.edge_type, data.node_type)
            print(f"   - Output shape: {logits.shape}")
            
    except Exception as e:
        print(f"❌ HAN model failed: {e}")
        return False
    
    # Test training
    print("\n🚀 Testing training...")
    try:
        import torch.optim as optim
        
        optimizer = optim.Adam(rgcn_model.parameters(), lr=0.01)
        trainer = Trainer(rgcn_model, optimizer, device=device)
        
        # Single training step
        loss = trainer.train_epoch(data)
        print(f"✅ Training step completed!")
        print(f"   - Loss: {loss:.4f}")
        
        # Validation step
        val_loss, val_acc = trainer.validate(data)
        print(f"   - Val Loss: {val_loss:.4f}")
        print(f"   - Val Acc: {val_acc:.4f}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False
    
    # Test evaluation
    print("\n📈 Testing evaluation...")
    try:
        evaluator = ModelEvaluator(
            model=rgcn_model,
            device=device,
            num_classes=stats["num_classes"],
        )
        
        results = evaluator.evaluate_model(data, split="test")
        metrics = results["metrics"]
        
        print(f"✅ Evaluation completed!")
        print(f"   - Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"   - Test F1 (Macro): {metrics['f1_macro']:.4f}")
        print(f"   - Test F1 (Micro): {metrics['f1_micro']:.4f}")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return False
    
    print("\n🎉 All tests passed! Implementation is working correctly.")
    return True


if __name__ == "__main__":
    success = test_implementation()
    sys.exit(0 if success else 1)
