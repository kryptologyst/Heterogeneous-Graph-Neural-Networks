"""Streamlit demo for heterogeneous graph neural networks."""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from pyvis.network import Network
import tempfile
import os

from src.data import HeterogeneousGraphDataset
from src.models import RGCN, HAN, HeterogeneousGNNClassifier
from src.utils import get_device, set_seed


def load_model_and_data(model_name: str, dataset_name: str):
    """Load model and data for demo."""
    # Set seed for reproducibility
    set_seed(42)
    
    # Load dataset
    dataset = HeterogeneousGraphDataset(name=dataset_name, root="data")
    data = dataset.get_data()
    data_stats = dataset.get_stats()
    
    # Create model
    if model_name == "RGCN":
        backbone = RGCN(
            in_channels=data_stats["num_features"],
            hidden_channels=64,
            out_channels=64,
            num_relations=data_stats["num_edge_types"],
            num_layers=2,
            dropout=0.5,
        )
    elif model_name == "HAN":
        backbone = HAN(
            in_channels=data_stats["num_features"],
            hidden_channels=64,
            out_channels=64,
            num_node_types=data_stats["num_node_types"],
            num_edge_types=data_stats["num_edge_types"],
            num_layers=2,
            heads=4,
            dropout=0.5,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = HeterogeneousGNNClassifier(
        backbone=backbone,
        num_classes=data_stats["num_classes"],
        dropout=0.5,
    )
    
    # Load pretrained weights if available
    model_path = f"checkpoints/{model_name.lower()}_{dataset_name}_best.pt"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        st.success(f"Loaded pretrained {model_name} model")
    else:
        st.warning(f"No pretrained model found at {model_path}. Using random weights.")
    
    return model, data, data_stats


def create_network_visualization(data, max_nodes=100):
    """Create interactive network visualization."""
    # Sample nodes if graph is too large
    if data.num_nodes > max_nodes:
        node_indices = torch.randperm(data.num_nodes)[:max_nodes]
        node_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        node_mask[node_indices] = True
        
        # Filter edges
        edge_mask = node_mask[data.edge_index[0]] & node_mask[data.edge_index[1]]
        filtered_edges = data.edge_index[:, edge_mask]
        
        # Create mapping from original to filtered indices
        node_mapping = {int(idx): i for i, idx in enumerate(node_indices)}
        
        # Remap edge indices
        filtered_edges = torch.tensor([
            [node_mapping[int(src)], node_mapping[int(dst)]]
            for src, dst in filtered_edges.t().tolist()
        ]).t()
        
        filtered_edge_types = data.edge_type[edge_mask]
        filtered_node_types = data.node_type[node_indices]
        filtered_features = data.x[node_indices]
        filtered_labels = data.y[node_indices] if hasattr(data, 'y') else None
    else:
        filtered_edges = data.edge_index
        filtered_edge_types = data.edge_type
        filtered_node_types = data.node_type
        filtered_features = data.x
        filtered_labels = data.y if hasattr(data, 'y') else None
    
    # Create NetworkX graph
    G = nx.Graph()
    
    # Add nodes
    for i in range(filtered_edges.size(1)):
        src, dst = filtered_edges[0, i].item(), filtered_edges[1, i].item()
        edge_type = filtered_edge_types[i].item()
        
        if not G.has_edge(src, dst):
            G.add_edge(src, dst, edge_type=edge_type)
    
    # Add node attributes
    for i in range(len(filtered_features)):
        G.add_node(i, 
                  node_type=filtered_node_types[i].item(),
                  features=filtered_features[i].tolist(),
                  label=filtered_labels[i].item() if filtered_labels is not None else 0)
    
    # Create PyVis network
    net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
    
    # Add nodes
    for node in G.nodes():
        node_data = G.nodes[node]
        node_type = node_data['node_type']
        label = node_data['label']
        
        # Color by node type
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        color = colors[node_type % len(colors)]
        
        net.add_node(node, 
                    label=f"Node {node}",
                    color=color,
                    title=f"Type: {node_type}, Label: {label}")
    
    # Add edges
    for edge in G.edges():
        edge_data = G.edges[edge]
        edge_type = edge_data['edge_type']
        
        # Color by edge type
        edge_colors = ['#FF9999', '#99FF99', '#9999FF', '#FFFF99', '#FF99FF']
        color = edge_colors[edge_type % len(edge_colors)]
        
        net.add_edge(edge[0], edge[1], color=color, title=f"Type: {edge_type}")
    
    # Configure physics
    net.set_options("""
    var options = {
      "physics": {
        "enabled": true,
        "stabilization": {"iterations": 100}
      }
    }
    """)
    
    return net


def analyze_node(model, data, node_idx):
    """Analyze a specific node."""
    model.eval()
    device = get_device()
    model.to(device)
    data = data.to(device)
    
    with torch.no_grad():
        if hasattr(data, 'node_type'):
            logits = model(data.x, data.edge_index, data.edge_type, data.node_type)
        else:
            logits = model(data.x, data.edge_index, data.edge_type)
        
        # Get predictions
        node_logits = logits[node_idx]
        node_probs = torch.softmax(node_logits, dim=0)
        node_pred = node_logits.argmax().item()
        
        # Get neighbors
        neighbors = data.edge_index[1][data.edge_index[0] == node_idx]
        neighbor_types = data.edge_type[data.edge_index[0] == node_idx]
        
        return {
            'logits': node_logits.cpu().numpy(),
            'probabilities': node_probs.cpu().numpy(),
            'prediction': node_pred,
            'neighbors': neighbors.cpu().numpy(),
            'neighbor_types': neighbor_types.cpu().numpy(),
        }


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Heterogeneous GNN Demo",
        page_icon="🕸️",
        layout="wide"
    )
    
    st.title("🕸️ Heterogeneous Graph Neural Networks Demo")
    st.markdown("Explore heterogeneous graphs and GNN predictions interactively!")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    model_name = st.sidebar.selectbox(
        "Model",
        ["RGCN", "HAN"],
        help="Choose the GNN model architecture"
    )
    
    dataset_name = st.sidebar.selectbox(
        "Dataset",
        ["synthetic", "aifb", "am", "mutag"],
        help="Choose the dataset"
    )
    
    # Load model and data
    try:
        model, data, data_stats = load_model_and_data(model_name, dataset_name)
        
        # Display dataset statistics
        st.sidebar.header("Dataset Statistics")
        st.sidebar.metric("Nodes", data_stats["num_nodes"])
        st.sidebar.metric("Edges", data_stats["num_edges"])
        st.sidebar.metric("Features", data_stats["num_features"])
        st.sidebar.metric("Classes", data_stats["num_classes"])
        st.sidebar.metric("Edge Types", data_stats["num_edge_types"])
        st.sidebar.metric("Node Types", data_stats["num_node_types"])
        
        # Main content
        tab1, tab2, tab3, tab4 = st.tabs(["Graph Visualization", "Node Analysis", "Model Performance", "Data Exploration"])
        
        with tab1:
            st.header("Graph Visualization")
            
            max_nodes = st.slider("Max nodes to display", 10, min(500, data_stats["num_nodes"]), 100)
            
            if st.button("Generate Visualization"):
                with st.spinner("Creating network visualization..."):
                    net = create_network_visualization(data, max_nodes)
                    
                    # Save to temporary file
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
                        net.save_graph(f.name)
                        
                        # Read and display
                        with open(f.name, 'r') as html_file:
                            html_content = html_file.read()
                        
                        st.components.v1.html(html_content, height=600)
                        
                        # Clean up
                        os.unlink(f.name)
        
        with tab2:
            st.header("Node Analysis")
            
            node_idx = st.number_input(
                "Node Index",
                min_value=0,
                max_value=data_stats["num_nodes"] - 1,
                value=0,
                help="Select a node to analyze"
            )
            
            if st.button("Analyze Node"):
                with st.spinner("Analyzing node..."):
                    analysis = analyze_node(model, data, node_idx)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Prediction")
                        st.metric("Predicted Class", analysis['prediction'])
                        
                        # Probability distribution
                        prob_df = pd.DataFrame({
                            'Class': range(len(analysis['probabilities'])),
                            'Probability': analysis['probabilities']
                        })
                        
                        fig = px.bar(prob_df, x='Class', y='Probability', 
                                    title='Class Probabilities')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.subheader("Neighbors")
                        st.metric("Number of Neighbors", len(analysis['neighbors']))
                        
                        if len(analysis['neighbors']) > 0:
                            neighbor_df = pd.DataFrame({
                                'Neighbor': analysis['neighbors'],
                                'Edge Type': analysis['neighbor_types']
                            })
                            st.dataframe(neighbor_df)
        
        with tab3:
            st.header("Model Performance")
            
            if st.button("Evaluate Model"):
                with st.spinner("Evaluating model..."):
                    from src.eval import ModelEvaluator
                    
                    evaluator = ModelEvaluator(
                        model=model,
                        device=get_device(),
                        num_classes=data_stats["num_classes"]
                    )
                    
                    # Evaluate on test set
                    results = evaluator.evaluate_model(data, split="test")
                    metrics = results["metrics"]
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                        st.metric("F1 (Macro)", f"{metrics['f1_macro']:.4f}")
                    
                    with col2:
                        st.metric("Precision (Macro)", f"{metrics['precision_macro']:.4f}")
                        st.metric("Recall (Macro)", f"{metrics['recall_macro']:.4f}")
                    
                    with col3:
                        st.metric("F1 (Micro)", f"{metrics['f1_micro']:.4f}")
                        if 'auc_roc_ovr' in metrics:
                            st.metric("AUC-ROC", f"{metrics['auc_roc_ovr']:.4f}")
        
        with tab4:
            st.header("Data Exploration")
            
            # Feature analysis
            st.subheader("Node Features")
            feature_df = pd.DataFrame(data.x.numpy())
            st.dataframe(feature_df.describe())
            
            # Edge type distribution
            st.subheader("Edge Type Distribution")
            edge_type_counts = torch.bincount(data.edge_type).numpy()
            edge_type_df = pd.DataFrame({
                'Edge Type': range(len(edge_type_counts)),
                'Count': edge_type_counts
            })
            
            fig = px.bar(edge_type_df, x='Edge Type', y='Count', 
                        title='Edge Type Distribution')
            st.plotly_chart(fig, use_container_width=True)
            
            # Node type distribution
            st.subheader("Node Type Distribution")
            node_type_counts = torch.bincount(data.node_type).numpy()
            node_type_df = pd.DataFrame({
                'Node Type': range(len(node_type_counts)),
                'Count': node_type_counts
            })
            
            fig = px.bar(node_type_df, x='Node Type', y='Count', 
                        title='Node Type Distribution')
            st.plotly_chart(fig, use_container_width=True)
            
            # Label distribution
            if hasattr(data, 'y'):
                st.subheader("Label Distribution")
                label_counts = torch.bincount(data.y).numpy()
                label_df = pd.DataFrame({
                    'Label': range(len(label_counts)),
                    'Count': label_counts
                })
                
                fig = px.bar(label_df, x='Label', y='Count', 
                            title='Label Distribution')
                st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error loading model/data: {str(e)}")
        st.info("Please make sure you have trained a model first by running the training script.")


if __name__ == "__main__":
    main()
