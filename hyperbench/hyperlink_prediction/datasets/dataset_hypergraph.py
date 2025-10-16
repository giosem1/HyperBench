import torch
from abc import ABC
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.hypergraph_data import HyperGraphData

class DatasetHyperGraph(InMemoryDataset, ABC):
    """ A class which obtains the the edge_index and time_saved from a dataset

        Args:
            dataset_name (string): The dataset's name.
            edge_index (Tensor): A tensor where are saved the node's id.
            time_saved (Tensor): A tensor where are saved the timestamp.
            nvert_attribute (Tensor): A tensor where are saved the node's attribute.
            root (string, optional): The folder's name where the dataset will be saved.
                 (default: 'data')
    """

    def __init__(self, dataset_name: str, root: str = 'data', *args, **kwargs):
        self.dataset_name = dataset_name
        super().__init__(root, *args, **kwargs)
        self.load(self.processed_paths[0])
        
    @property
    def raw_dir(self):
        return f"{self.root}/{self.dataset_name}/raw"

    @property
    def processed_dir(self):
        return f"{self.root}/{self.dataset_name}/processed"

    @property
    def raw_file_names(self):
        return [self.dataset_name + "-" + pfx + ".txt" for pfx in ["node-labels", "nverts", "simplices", "times"]]

    @property
    def processed_file_names(self):
        return "processed.pt"
    
    def __getitem__(self, idx) -> HyperGraphData:
        """ Take a index and find in the tensor the index of the node's number
            which is associeted at the list of nodes's id.

            Args: 
                index (int): the index of the node in the tensor edge_index
                return: the tuple containing as first element a tensor which contains a list of nodes's id 
                    from the file which finish with '-simplices.txt'
                    and a list of the index of the node's number,
                    and as second element the time which is contained in the list of the time_stampded.
        """        
        if isinstance(idx, int):
            idx = torch.tensor([idx])
        if isinstance(idx, list):
            idx = torch.tensor(idx)
            
        edge_index_mask = torch.isin(self._data.edge_index[1], idx)
        edge_index = self._data.edge_index[:, edge_index_mask]
        _, edge_index[1] = edge_index[1].unique(return_inverse=True)
        return HyperGraphData(
            x=self._data.x,
            edge_index=edge_index,
            edge_attr=self._data.edge_attr[idx],
            num_nodes=self._data.num_nodes,
            num_edges=edge_index[1].max().item() + 1,
        )
    
    def __len__(self) -> int:
        """Return the number of the node in the hypergraph"""
        return self._data.edge_index[1].max().item() + 1