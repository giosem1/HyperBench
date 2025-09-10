from abc import ABC
import torch
from torch import Tensor

class HyperlinkPredictionResult(ABC):

    def __init__(self, 
                 p_edge_index: Tensor, 
                 n_edge_index: Tensor, 
                 device="cpu"):
        self.device = device
        self.__p_edge_index = p_edge_index.to(device)
        self.__n_edge_index = n_edge_index.to(device)
        
        _, self.__p_edge_index[1] = torch.unique(self.__p_edge_index[1], return_inverse=True)
        _, self.__n_edge_index[1] = torch.unique(self.__n_edge_index[1], return_inverse=True)

    @property
    def p_edge_index(self) -> Tensor:
        return self.__p_edge_index

    @property
    def n_edge_index(self) -> Tensor:
        return self.__n_edge_index

    @property
    def num_p_edges(self):
        return torch.unique(self.__p_edge_index[1]).shape[0]

    @property
    def num_n_edges(self):
        return torch.unique(self.__n_edge_index[1]).shape[0]

    @property
    def edge_index(self) -> Tensor:
        max_index = torch.max(self.p_edge_index[1]) + 1
        n_edge_index = self.n_edge_index.clone()
        n_edge_index[1] += max_index
        return torch.hstack([self.p_edge_index, n_edge_index])

    @property
    def y(self) -> Tensor:
        y_p = torch.ones((self.num_p_edges, 1), device=self.device)
        y_n = torch.zeros((self.num_n_edges, 1), device=self.device)
        return torch.vstack([y_p, y_n])

    def __repr__(self):
        return self.edge_index.__repr__()
