import argparse
def execute():
    parser = argparse.ArgumentParser(description="Insert dataset_name, insert negative_sampling method")
    parser.add_argument('--dataset_name', type=str, help="The dataset's name, possible dataset's name: \nIMDB,\nCURSERA,\nARXIV", required=True)
    parser.add_argument('--negative_sampling', type=str, help="negative sampling method to use, possible methods: \n SizedHypergraphNegativeSampler,\nMotifHypergraphNegativeSampler,\nCliqueHypergraphNegativeSampler", required=True)
    parser.add_argument('--hlp_method', type=str, help="hyperlink prediction method to use, possible method: \nCommonNeighbors", required=True)
    args = parser.parse_args()
    dataset_name= args.dataset_name
    negative_method = args.negative_sampling
    hlp_method = args.hlp_method

    import torch
    import numpy as np
    import seaborn as sns
    import torch.nn as nn
    import matplotlib.pyplot as plt
    from random import randint
    from hyperlink_prediction.loader.dataloader import DatasetLoader
    from hyperlink_prediction.hyperlink_prediction_algorithm import CommonNeighbors
    from hyperlink_prediction.datasets.imdb_dataset import CHLPBaseDataset, IMDBHypergraphDataset, ARXIVHypergraphDataset, COURSERAHypergraphDataset
    from utils.set_negative_samplig_method import setNegativeSamplingAlgorithm
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
    
    dataset : CHLPBaseDataset
    match(dataset_name):
        case 'IMDB': 
            dataset = IMDBHypergraphDataset("./data", pre_transform= pre_transform)
        case 'ARXIV':
            dataset = ARXIVHypergraphDataset("./data", pre_transform = pre_transform)
        case 'COURSERA':
            dataset = COURSERAHypergraphDataset("./data", pre_transform = pre_transform)

    train_dataset, test_dataset, _, _, _, _ = train_test_split(dataset, test_size = 0.4)

    loader = DatasetLoader(
        dataset, 
        negative_method, 
        dataset._data.num_nodes,
        batch_size=4000, 
        shuffle=True, 
        drop_last = True
        )

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

    negative_hypergraph = setNegativeSamplingAlgorithm(negative_method,test_dataset.x.__len__()).generate(test_dataset._data.edge_index)
    edge_index_test = test_dataset._data.edge_index.clone()
    test_dataset.y = torch.vstack((
        torch.ones((test_dataset._data.edge_index[1].max() + 1, 1), device= test_dataset.x.device),
        torch.zeros((edge_index_test[1].max() + 1, 1), device= test_dataset.x.device)
    ))

    test_dataset_ = HyperGraphData(
        x = test_dataset.x,
        edge_index= negative_hypergraph.edge_index,
        edge_attr= torch.vstack((test_dataset.edge_attr, test_dataset.edge_attr)),
        y = test_dataset.y,
        num_nodes = test_dataset._data.num_nodes
    )

    for epoch in trange(150):
        model.train()
        optimizer.zero_grad()
        for i, h in tqdm(enumerate(loader), leave = False):
            h = h.to(device)
            negative_sampler = setNegativeSamplingAlgorithm(negative_method, h.num_nodes)
            negative_test = negative_sampler.generate(h.edge_index)

            hlp_method = CommonNeighbors(h.num_nodes)
            hlp_result = hlp_method.generate(negative_test.edge_index)

            y_pos = torch.ones(hlp_result.edge_index.size(1), 1, device=h.x.device)
            y_neg = torch.zeros(negative_test.edge_index.size(1), 1, device=h.x.device)

            combined_edge_index = torch.hstack([hlp_result.edge_index, negative_test.edge_index])
            combined_y = torch.vstack([y_pos, y_neg])

            pos_edge_attr = torch.ones((hlp_result.edge_index.size(1), h.edge_attr.size(1)), device=h.x.device)
            neg_edge_attr = torch.zeros((negative_test.edge_index.size(1), h.edge_attr.size(1)), device=h.x.device)
            combined_edge_attr = torch.vstack([pos_edge_attr, neg_edge_attr])

            h_ = HyperGraphData(
                x=h.x,
                edge_index=combined_edge_index.to(device),
                edge_attr=combined_edge_attr,
                y=combined_y.to(device),
                num_nodes=h.num_nodes
            )

            y_train = model(h_.x, h_.edge_attr, h_.edge_index)
            if y_train.size(1) != 1:
                y_train = y_train[:, 0:1]
            loss = criterion(y_train, h_.y)
            loss.backward()
            writer.add_scalar("Loss/train", loss.item(), epoch * len(loader) + i)
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            y_test = model(test_dataset_.x.to(device), test_dataset_.edge_attr.to(device), test_dataset_.edge_index.to(device))
            y_test = torch.sigmoid(y_test)
            if y_test.size(1) != 1:
                y_test = y_test[:, 0:1]
            loss = test_criterion(y_test, test_dataset_.y)
            writer.add_scalar("Loss/test", loss.item(), epoch)
            roc_auc = roc_auc_score(test_dataset_.y.cpu().numpy(), y_test.cpu().numpy())
            writer.add_scalar("ROC_AUC/test", roc_auc, epoch)
        
    cutoff = sensivity_specifivity_cutoff(test_dataset_.y.cpu().numpy(), y_test.cpu().numpy())
    cm = confusion_matrix(
        test_dataset_.y.cpu().numpy(),
        (y_test > cutoff).cpu().numpy(),
        labels=[0, 1],
        normalize='true'
    )

    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Normalized Confusion Matrix")

    plt.show()

    negative_hypergraph = setNegativeSamplingAlgorithm(negative_method, test_dataset.x.__len__()).generate(test_dataset._data.edge_index)
    test_dataset.y = torch.vstack((
        torch.ones((test_dataset._data.edge_index[1].max() + 1, 1), device=test_dataset.x.device),
        torch.zeros((test_dataset._data.edge_index[1].max() + 1, 1), device= test_dataset.x.device)
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