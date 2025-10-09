import torch
from torch import Tensor
from .hyperlink_prediction_base import HyperlinkPredictor
from .hyperlink_prediction_result import HyperlinkPredictionResult

class CommonNeighbors(HyperlinkPredictor):
    def __init__(self, device='cpu'):
        super().__init__(device)
        self.H = None        
        self.num_node = None
        self.num_hyperlink = None

    def fit(self, X, y, edge_index, *args, **kwargs):
        
        self.num_node = int(edge_index[0].max().item()) + 1
        self.num_hyperlink = int(edge_index[1].max().item()) + 1

        sparse = torch.sparse_coo_tensor(
            edge_index,
            torch.ones(edge_index.shape[1], device=self.device),
            (self.num_node, edge_index.max().item() + 1),
            device=self.device
        )

        self.H = sparse.to_dense()
        return self 
    
    def score_CN(self, H, u, v):
        return torch.dot(H[u], H[v]).item()
    
    def predict(self, edge_index: Tensor):
        if self.H is None:
            if edge_index is None:
                raise ValueError("Model not fitted. Call fit() first.")
            self.fit(None, None, edge_index)
        H = self.H
        
        CN_matrix = torch.matmul(H, H.T)

        new_edges = torch.nonzero(torch.triu(CN_matrix, diagonal=1)).T 

        return HyperlinkPredictionResult(
            edge_index=new_edges,
            device=self.device
        )
