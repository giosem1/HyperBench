import torch
import numpy as np
from abc import abstractmethod
from hyperlink_prediction_result import HyperlinkPredictionResult
class HyperlinkPredictor():
    
    def __init__(self, num_node: int, device: torch.device = torch.device('cpu')):
        self.num_node = num_node
        self.device = device

    @abstractmethod
    def fit(self, X, y, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, X, edge_index: torch.Tensor) -> HyperlinkPredictionResult:
        pass

