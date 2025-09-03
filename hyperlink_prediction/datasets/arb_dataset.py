import gdown
import tarfile
import torch
import torch.nn.functional as F
from hyperlink_prediction.datasets.dataset_hypergraph import DatasetHyperGraph
from torch_geometric.data.hypergraph_data import HyperGraphData
from os import remove, listdir

class ARBDataset(DatasetHyperGraph): 
    """A object that save a hypergraph dataset from Google Drive and load in memory

        Args:
            dataset_name (string): The dataset's name.
            root (string): The root where the dataset will be saved.
                 (default: 'datasets')
    """

    GDRIVE_IDs = {
        'coauth-DBLP': '15YpIK8vvzQJXyQC4bt-Sz951e4eb0rc_',
        "coauth-MAG-Geology": "14MOWsEJyNGiFKumvmgMcU1CqekDBayYV",
        "email-Enron": "1tTVZkdpgRW47WWmsrdUCukHz0x2M6N77",
        "tags-math-sx": "1eDevpF6EZs19rLouNpiKGLIlFOLUfKKG",
        "contact-high-school": "1VA2P62awVYgluOIh1W4NZQQgkQCBk-Eu",
        "contact-primary-school": "1sBHSEIyvVKavAho524Ro4cKL66W6rn-t",
        "NDC-substances": "1dLJt3qzAOYieay03Sp9h8ZfVMiU-nMqC"
    }
    

    def __init__(self, dataset_name: str, *args, **kwargs):
        self.dataset_name = dataset_name
        super().__init__(dataset_name , *args, **kwargs)

    def download(self) -> None:
        """ Take the dataset from Google Drive through the name of dataset,
            which is associeted the key, and download it in memory in the folder
            datasets.
        """
        if len(listdir(self.raw_dir)) > 0:
            return
        
        archive_file_name = self.raw_dir + "/" + "raw.tar.gz"
        gdown.download(id=ARBDataset.GDRIVE_IDs[self.dataset_name], output=archive_file_name)
        with tarfile.open(archive_file_name) as tar:
            members = list(filter(lambda s: "/" + self.dataset_name in str(s), tar.getmembers()))
            tar.extractall(self.raw_dir, members)
        remove(archive_file_name)

    def process(self) -> None:
        """ Process the files of the verts and the simplices in a tensor containing two list
            the first contain the nodes's id
            and the second the index of the nodes's numbers
            and then serialize the tensor 
        """
        edge_index = []
        edge_offsets = []
        with open(self.raw_dir + f"/{self.dataset_name}/{self.dataset_name}-nverts.txt", "r") as f:
            with open(self.raw_dir + f"/{self.dataset_name}/{self.dataset_name}-simplices.txt", "r") as g:
                for i, nverts in enumerate(f):
                    edge_offsets.append(len(edge_index))
                    for _ in range(int(nverts)):
                        node = int(g.readline()) - 1
                        edge_index.append((node, i))
        edge_offsets.append(len(edge_index))
        edge_index = torch.tensor(edge_index).T
        edge_offsets = torch.tensor(edge_offsets)
        with open(self.raw_dir + f"/{self.dataset_name}/{self.dataset_name}-times.txt", "r") as tf:
            times = torch.tensor([int(i) for i in tf]).unsqueeze(1)

        _, edge_index[0] = edge_index[0].unique(return_inverse=True)
        num_nodes = edge_index[0].max().item() + 1 
        data_list = [HyperGraphData(
            x=torch.eye(num_nodes),
            edge_index=edge_index,
            edge_attr=times,
            num_nodes=num_nodes,
        )]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])

    def unique_hyperedges_edge_index(edge_index: torch.Tensor,
                                 num_nodes: int,
                                 num_edges: int,
                                 return_counts: bool = False,
                                 return_duplicates: bool = False) -> torch.Tensor:
        """ Process the edge_index of the hypergraph and return a tensor which have
            no repetition of the edege...
        """
        A = torch.sparse_coo_tensor(
            indices=edge_index,
            values=torch.ones(edge_index.size(1), dtype=torch.float, device=edge_index.device),
            size=(num_nodes, num_edges)
        )
        B = A.T @ A
        h_degrees = edge_index[1].bincount()
        duplicates = torch.logical_and(
            B.indices()[0] < B.indices()[1],
            torch.logical_and(
                B.values() == h_degrees[B.indices()[0]],
                B.values() == h_degrees[B.indices()[1]]
            )
        )
        duplicate_edges = B.indices()[1, duplicates].unique()
        unique_edge_index = edge_index[:,
            ~torch.isin(
                edge_index[1],
                duplicate_edges
        )]
        _ , unique_edge_index[1] = torch.unique(unique_edge_index[1], return_inverse=True)
        if return_counts:
            counts = B.indices()[0, duplicates][
            ~torch.isin(
                    B.indices()[0, duplicates],
                    B.indices()[1, duplicates]
            )].bincount()
            counts = F.pad(counts, (0, num_edges - counts.size(0)))
            unique_hyperedges = torch.unique(edge_index[1])
            counts[unique_hyperedges] += 1
            if return_duplicates:
                return unique_edge_index, counts, duplicate_edges
            return unique_edge_index, counts
        if return_duplicates:
            return unique_edge_index, duplicate_edges
        return unique_edge_index

    def pre_transform(self, data):
        edge_index, counts = self.unique_hyperedges_edge_index(data.edge_index,
                                    data.num_nodes,
                                    data.num_edges,
                                    return_counts=True)
        num_edges = edge_index[1].max() + 1
        return HyperGraphData(
            x=data.x,
            edge_index=edge_index,
            edge_attr=data.edge_attr,
            num_nodes=data.num_nodes,
            num_edges=num_edges
        )
    
torch.serialization.add_safe_globals([HyperGraphData])
