import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.hypergraph_data import HyperGraphData

class SplitDataset(InMemoryDataset):
    """ A class that split the dataset in train and test."""
    def __getitem__(self, idx) -> HyperGraphData:
        if isinstance(idx, int):
            idx = torch.tensor([idx])
        if isinstance(idx, list):
            idx = torch.tensor(idx)
        edge_index_mask = torch.isin(self._data.edge_index[1], idx)
        edge_index = self._data.edge_index[:, edge_index_mask]
        _, edge_index[1] = edge_index[1].unique(return_inverse = True)
        return HyperGraphData(
            x = self._data.x,
            edge_index=edge_index,
            edge_attr=self._data.edge_attr[idx]
        )
    
    def __len__(self) -> int:
        return self._data.edge_index[1].max().item() + 1

def train_test_split(dataset: InMemoryDataset, test_size: float = 0.2):
    indices = torch.randperm(len(dataset), device=dataset.x.device)
    split = int(len(dataset) * (1 - test_size))
    train_indices = torch.sort(indices[:split]).values
    test_indices = torch.sort(indices[split:]).values
    train_data = dataset[train_indices]

    class TrainDataset(SplitDataset):
        """The class of the Train Dataset"""
        def __init__(self):
            super().__init__()
            self._data = train_data
    train_dataset = TrainDataset()
    
    test_data = dataset[test_indices]
    class TestDataset(SplitDataset):
        """The class of the Test Dataset"""
        def __init__(self):
            super().__init__()
            self._data = test_data
    test_dataset = TestDataset()

    train_mask = torch.zeros(len(dataset), dtype= torch.bool, device= dataset.x.device)
    train_mask[train_indices] = True
    test_mask = torch.zeros(len(dataset), dtype= torch.bool, device= dataset.x.device)
    test_mask[test_indices] = True
    return train_dataset, test_dataset, train_indices, test_indices, train_mask, test_mask