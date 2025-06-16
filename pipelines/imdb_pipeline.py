import torch
import argparse
import numpy as np
import seaborn as sns
import torch.nn as nn
from random import randint
from hyperlink_prediction.loader.dataloader import DatasetLoader
from hyperlink_prediction.datasets.imdb_dataset import IMDBHypergraphDataset
from negative_sampling.hypergraph_negative_sampling_algorithm import MotifHypergraphNegativeSampler
from utils.hyperlink_train_test_split import train_test_split
from torch_geometric.nn import HypergraphConv
from tqdm.auto import trange, tqdm
from torch_geometric.data.hypergraph_data import HyperGraphData
from torch_geometric.nn.aggr import MeanAggregation
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

def sensivity_specifivity_cutoff(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    idx = np.argmax( tpr - fpr)

    return thresholds[idx]

writer = SummaryWriter(f"./logs/{randint(0,10000)}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def pre_transform(data: HyperGraphData):
    data.edge_index = data.edge_index[:, torch.isin(data.edge_index[1], (data.edge_index[1].bincount() > 1).nonzero())]
    unique, inverse = data.edge_index[1].unique(return_inverse = True)
    data.edge_attr = data.edge_attr[unique]
    data.edge_index[1] = inverse

    return data

dataset  = IMDBHypergraphDataset("./data", pre_transform= pre_transform)
train_dataset, test_dataset, _, _, _, _ = train_test_split(dataset, test_size = 0.4)

loader = DatasetLoader(dataset, MotifHypergraphNegativeSampler(dataset._data.num_nodes),batch_size=4000, shuffle=True, drop_last = True)

class Model(nn.Module):
    
    def __init__(self, 
                in_channels: int,
                hidden_channels: int,
                out_channels: int,
                num_layers: int = 1):
        super(Model, self).__init__()
        self.dropout = nn.Dropout(0.3)
        self.activation = nn.LeakyReLU()
        self.in_norm = nn.LayerNorm(in_channels)
        self.in_proj = nn.Linear(in_channels, hidden_channels)
        self.e_proj = nn.Linear(in_channels, hidden_channels)
        self.e_norm = nn.LayerNorm(in_channels)

        for i in range(num_layers):
            setattr(self, f"n_norm_{i}", nn.LayerNorm(hidden_channels))
            setattr(self, f"e_norm_{i}", nn.LayerNorm(hidden_channels))
            setattr(self, f"hgconv_{i}",HypergraphConv(
                hidden_channels,
                hidden_channels,
                use_attention=True,
                concat=False,
                heads=1
            ))
            setattr(self, f"skip_{i}", nn.Linear(hidden_channels, hidden_channels))
        self.num_layers = num_layers

        self.aggr = MeanAggregation()
        self.linear = nn.Linear(hidden_channels, hidden_channels)
    
    def forward(self, x, x_e, edge_index):
        x = self.in_norm(x)
        x = self.in_proj(x)
        x = self.activation(x)
        x = self.dropout(x)

        x_e = self.e_norm(x_e)
        x_e = self.e_proj(x_e)
        x_e = self.activation(x_e)
        x_e = self.dropout(x_e)

        for i in range(self.num_layers):
            n_norm = getattr(self, f"n_norm_{i}")
            e_norm = getattr(self, f"e_norm_{i}")
            hgconv = getattr(self, f"hgconv_{i}")
            skip = getattr(self, f"skip_{i}")
            x = n_norm(x)
            x_e = e_norm(x_e)
            x = self.activation(hgconv(x, edge_index, hyperedge_attr = x_e)) + skip(x)

        x = self.aggr(x[edge_index[0]], edge_index[1])
        x = self.linear(x)

        return x
    
model = Model(
    in_channels = dataset.num_features,
    hidden_channels = 256,
    out_channels= 1
    ).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.BCEWithLogitsLoss()
test_criterion = torch.nn.BCELoss()

negative_hypergraph = MotifHypergraphNegativeSampler(test_dataset.x.__len__()).generate(test_dataset.edge_index)
edge_index_test = test_dataset.edge_index.clone()
test_dataset.y = torch.vstack((
    torch.ones((test_dataset.edge_index[1].max() + 1, 1), device= test_dataset.x.device),
    torch.zeros((edge_index_test[1].max() + 1, 1), device= test_dataset.x.device)
))

test_dataset_ = HyperGraphData(
    x = test_dataset.x,
    edge_index= negative_hypergraph.edge_index,
    edge_attr= torch.vstack((test_dataset.edge_attrs, test_dataset.edge_attrs)),
    y = test_dataset.y,
    num_nodes = test_dataset.num_nodes
)

for epoch in trange(150):
    model.train()
    optimizer.zero_grad()
    for i, h in tqdm(enumerate(loader), leave = False):
        h = h.to(device)
        negative_test = MotifHypergraphNegativeSampler(h.x.__len__()).generate(h.edge_index)
        edge_index = h.edge_index.clone()
        h.y = torch.vstack((
            torch.ones((h.edge_index[1].max() + 1, 1), device= h.x.device),
            torch.zeros((edge_index[1].max() + 1, 1), device= h.x.device)
        ))

        h_ = HyperGraphData(
            x = h.x,
            edge_index= negative_test.edge_index,
            edge_attr= torch.vstack((h.edge_attr, h.edge_attr)),
            y = h.y,
            num_nodes = negative_test.num_edges
        )

        y_train = model(h_.x, h_.edge_attr, h_.edge_index)
        loss = criterion(y_train, h_.y)
        loss.backward()
        writer.add_scalar("Loss/train", loss.item(), epoch * len(loader) + i)
    optimizer.step()
    
    model.eval()
    with torch.no_grad():
        y_test = model(test_dataset_.x.to(device), test_dataset_.edge_attr.to(device), test_dataset_.edge_index.to(device))
        y_test = torch.sigmoid(y_test)
        loss = test_criterion(y_test, test_dataset_.y)
        writer.add_scalar("Loss/test", loss.item(), epoch)
        roc_auc = roc_auc_score(test_dataset_.y.cpu().numpy(), y_test.cpu().numpy())
        writer.add_scalar("ROC_AUC/test", roc_auc, epoch)
    
cutoff = sensivity_specifivity_cutoff(test_dataset_.y.cpu().numpy(), y_test.cpu().numpy())
cm = confusion_matrix(
    test_dataset_.y.cpu().numpy(),
    (y_test > cutoff).cpu().numpy(),
    labels = [0,1],
    normalize = True 
)

sns.heatmap(cm, annot= True, cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])

negative_hypergraph = MotifHypergraphNegativeSampler(test_dataset.x.__len__()).generate(test_dataset.edge_index)
test_dataset.y = torch.vstack((
    torch.ones((test_dataset.edge_index[1].max() + 1, 1), device=test_dataset.x.device),
    torch.zeros((edge_index[1].max() + 1, 1), device= test_dataset.x.device)
))

test_dataset_ = HyperGraphData(
    x = test_dataset.x,
    edge_index= negative_hypergraph.edge_index,
    edge_attr= torch.vstack((test_dataset.edge_attr, test_dataset.edge_attr)),
    y = test_dataset.y,
    num_nodes = negative_hypergraph.num_edges
)

y_test = model(test_dataset_.x.to(device), test_dataset_.edge_attr.to(device), test_dataset_.edge_index.to(device))
y_test = torch.sigmoid(y_test)