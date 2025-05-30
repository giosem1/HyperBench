import torch
import numpy as np
from abc import abstractmethod

class HypergraphNegativeSampler():
    """ A class Negative Sampler which samples the negative samples.

        Args:
            num_node (int): The hypergraph's number of the nodes.
    """
    def __init__(self, num_node: int):
        self.num_node = num_node
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
