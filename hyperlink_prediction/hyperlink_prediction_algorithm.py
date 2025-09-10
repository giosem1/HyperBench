import torch
import itertools
import torch_geometric.nn.aggr as aggr
from enum import Enum
from torch import Tensor
from .hyperlink_prediction_base import HypergraphSampler
from .hyperlink_prediction_result import HyperlinkPredictionResult

class CommonNeighbros(HypergraphSampler):

    def score_CN(self, H, u, v):
        return torch.dot(H[u], H[v]).item()
    
    def generate(self, edge_index: Tensor, negative_edge_index: Tensor = None):
        sparse = torch.sparse_coo_tensor(
            edge_index,
            torch.ones(edge_index.shape[1], device=self.device),
            (self.num_node, edge_index.max().item() + 1),
            device=self.device
        )
        H = sparse.to_dense()
        
        CN_matrix = torch.matmul(H, H.T)

        new_edges = torch.nonzero(torch.triu(CN_matrix, diagonal=1)).T 

        if negative_edge_index is None:
            negative_edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)

        return HyperlinkPredictionResult(
            p_edge_index=new_edges,
            n_edge_index=negative_edge_index,
            device=self.device
        )
