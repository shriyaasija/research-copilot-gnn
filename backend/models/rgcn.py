"""
R-GCN (Relational Graph Convolutional Network) model definition.

This module defines the R-GCN architecture used for learning node embeddings
in the heterogeneous citation graph. The model supports multiple relation types
(cites, mentions, wrote) and produces 128-dimensional node embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import RGCNConv
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False


class RGCN(nn.Module):
    """
    Relational Graph Convolutional Network for heterogeneous citation graphs.
    
    Architecture:
        - Input projection layer (input_dim -> hidden_dim)
        - 2 R-GCN layers with basis decomposition
        - Output projection (hidden_dim -> embedding_dim)
        - Layer normalization + dropout for regularization
    
    Args:
        input_dim: Dimension of input node features (384 for MiniLM embeddings)
        hidden_dim: Hidden layer dimension (default: 256)
        embedding_dim: Output embedding dimension (default: 128)
        num_relations: Number of edge relation types (default: 5)
        num_bases: Number of basis matrices for R-GCN decomposition (default: 3)
        dropout: Dropout rate (default: 0.3)
    """
    
    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 256,
        embedding_dim: int = 128,
        num_relations: int = 5,
        num_bases: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        if not HAS_TORCH_GEOMETRIC:
            raise ImportError(
                "torch-geometric is required for the R-GCN model. "
                "Install with: pip install torch-geometric"
            )
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_relations = num_relations
        self.dropout = dropout
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        
        # R-GCN layers
        self.conv1 = RGCNConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            num_relations=num_relations,
            num_bases=num_bases,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        self.conv2 = RGCNConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            num_relations=num_relations,
            num_bases=num_bases,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, embedding_dim)
        self.output_norm = nn.LayerNorm(embedding_dim)
    
    def forward(self, x, edge_index, edge_type):
        """
        Forward pass through the R-GCN.
        
        Args:
            x: Node feature matrix [num_nodes, input_dim]
            edge_index: Edge index tensor [2, num_edges]
            edge_type: Edge type tensor [num_edges]
        
        Returns:
            Node embeddings [num_nodes, embedding_dim]
        """
        # Input projection
        h = self.input_proj(x)
        h = self.input_norm(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # R-GCN layer 1
        h = self.conv1(h, edge_index, edge_type)
        h = self.norm1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # R-GCN layer 2
        h = self.conv2(h, edge_index, edge_type)
        h = self.norm2(h)
        h = F.relu(h)
        
        # Output projection
        embeddings = self.output_proj(h)
        embeddings = self.output_norm(embeddings)
        
        # L2 normalize for cosine similarity
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        return embeddings
    
    def get_embeddings(self, x, edge_index, edge_type):
        """Get normalized embeddings without gradient tracking."""
        self.eval()
        with torch.no_grad():
            return self.forward(x, edge_index, edge_type)


class ContrastiveRGCN(nn.Module):
    """
    R-GCN with contrastive learning objective.
    
    Uses a projection head on top of R-GCN embeddings for contrastive training,
    which is removed during inference to use the base embeddings.
    
    Args:
        rgcn: Base R-GCN model
        projection_dim: Dimension of the projection head output (default: 64)
    """
    
    def __init__(self, rgcn: RGCN, projection_dim: int = 64):
        super().__init__()
        self.rgcn = rgcn
        self.projection_head = nn.Sequential(
            nn.Linear(rgcn.embedding_dim, rgcn.embedding_dim),
            nn.ReLU(),
            nn.Linear(rgcn.embedding_dim, projection_dim),
        )
    
    def forward(self, x, edge_index, edge_type):
        """Get projected embeddings for contrastive loss."""
        embeddings = self.rgcn(x, edge_index, edge_type)
        projections = self.projection_head(embeddings)
        projections = F.normalize(projections, p=2, dim=-1)
        return projections
    
    def get_embeddings(self, x, edge_index, edge_type):
        """Get base R-GCN embeddings (without projection head)."""
        return self.rgcn.get_embeddings(x, edge_index, edge_type)
