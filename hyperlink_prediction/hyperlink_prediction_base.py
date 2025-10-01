import torch
import numpy as np
from abc import abstractmethod

class HypergraphSampler():
    
    def __init__(self, num_node: int, device: torch.device = torch.device('cpu')):
        self.num_node = num_node
        self.device = device

    @abstractmethod
    def fit(self, *args, **kwargs):
        pass

    @abstractmethod
    def generate(self, edge_index: torch.Tensor):
        pass
    
    @abstractmethod
    def transform(self, edge_index: np.ndarray):
        pass

    def trasform(self, edge_index: torch.Tensor):
        return self.generate(edge_index)
