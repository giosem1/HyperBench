import torch
from torch import Tensor
from .hyperlink_prediction_base import HypergraphSampler
from .hyperlink_prediction_result import HyperlinkPredictionResult

class CommonNeighbors(HypergraphSampler):

    def score_CN(self, H, u, v):
        return torch.dot(H[u], H[v]).item()
    
    def generate(self, edge_index: Tensor):
        sparse = torch.sparse_coo_tensor(
            edge_index,
            torch.ones(edge_index.shape[1], device=self.device),
            (self.num_node, edge_index.max().item() + 1),
            device=self.device
        )
        H = sparse.to_dense()
        
        CN_matrix = torch.matmul(H, H.T)

        new_edges = torch.nonzero(torch.triu(CN_matrix, diagonal=1)).T 

        return HyperlinkPredictionResult(
            edge_index=new_edges,
            device=self.device
        )
