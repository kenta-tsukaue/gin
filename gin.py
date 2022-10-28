import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset

from torch_geometric.utils import to_networkx

import networkx as nx
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from torch_geometric.utils import to_networkx
from mpl_toolkits.mplot3d import Axes3D

import numpy as np

# チェック関数(グラフ集合)
def check_graphs(data):
    print(f'Dataset: {data}')
    print('-------------------')
    print(f'Number of graphs: {len(data)}')
    print(f'Number of nodes: {data[0].x.shape[0]}')
    print(f'Number of features: {data.num_features}')
    print(f'Number of classes: {data.num_classes}')

# チェック関数(単一グラフ)
def check_graph(data):
    '''グラフ情報を表示'''
    print("グラフ構造:", data)
    print("グラフのキー: ", data.keys)
    print("ノード数:", data.num_nodes)
    print("エッジ数:", data.num_edges)
    print("ノードの特徴量数:", data.num_node_features)
    print("孤立したノードの有無:", data.contains_isolated_nodes())
    print("自己ループの有無:", data.contains_self_loops())
    print("====== ノードの特徴量:x ======")
    print(data['x'])
    print("====== ノードのクラス:y ======")
    print(data['y'])
    print("========= エッジ形状 =========")
    print(data['edge_index'])

# グラフ構造の可視化関数
def draw_graph(data):
    #networkxのグラフに変換
    nxg = to_networkx(data)
    #可視化のためのページランク計算
    pr = nx.pagerank(nxg)
    pr_max = np.array(list(pr.values())).max()

    #可視化する際のノード位置
    draw_pos = nx.spring_layout(nxg, seed=0)

    #ノードの色設定
    cmap = plt.get_cmap("tab10")
    labels = data.y.numpy()
    colors = [cmap(l) for l in labels]

    #図のサイズ
    plt.figure(figsize=(10, 10))

    #描画
    nx.draw_networkx_nodes(nxg,draw_pos, node_size=[v / pr_max * 1000 for v in pr.values()], node_color=colors, alpha=0.5)
    nx.draw_networkx_edges(nxg, draw_pos, arrowstyle='-', alpha=0.2)
    nx.draw_networkx_labels(nxg, draw_pos, font_size=10)

    plt.title('Protein')
    plt.show()

def draw_graph_3d(data):
    G = to_networkx(data, to_undirected=True)

    # 3D spring layout
    pos = nx.spring_layout(G, dim=3, seed=0)

    # Extract node and edge positions from the layout
    node_xyz = np.array([pos[v] for v in sorted(G)])
    edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges()])

    # Create the 3D figure
    fig = plt.figure(figsize=(16,16))
    ax = fig.add_subplot(111, projection="3d")

    # Suppress tick labels
    for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
        dim.set_ticks([])

    # Plot the nodes - alpha is scaled by "depth" automatically
    ax.scatter(*node_xyz.T, s=500, c="#0A047A")

    # Plot the edges
    for vizedge in edge_xyz:
        ax.plot(*vizedge.T, color="tab:gray")

    # fig.tight_layout()
    plt.show()

dataset = TUDataset(root='.', name='PROTEINS').shuffle()


#check_graphs(dataset)
#check_graph(dataset[0])
#draw_graph(dataset[0])

draw_graph_3d(dataset[0])


from torch_geometric.loader import DataLoader

# Create training, validation, and test sets
train_dataset = dataset[:int(len(dataset)*0.8)]
val_dataset   = dataset[int(len(dataset)*0.8):int(len(dataset)*0.9)]
test_dataset  = dataset[int(len(dataset)*0.9):]

print(f'Training set   = {len(train_dataset)} graphs')
print(f'Validation set = {len(val_dataset)} graphs')
print(f'Test set       = {len(test_dataset)} graphs')

# Create mini-batches
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print('\nTrain loader:')
for i, subgraph in enumerate(train_loader):
    print(f' - Subgraph {i}: {subgraph}')

print('\nValidation loader:')
for i, subgraph in enumerate(val_loader):
    print(f' - Subgraph {i}: {subgraph}')

print('\nTest loader:')
for i, subgraph in enumerate(test_loader):
    print(f' - Subgraph {i}: {subgraph}')


from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.nn import global_mean_pool, global_add_pool


class GCN(torch.nn.Module):
    """GCN"""
    def __init__(self, dim_h):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, dim_h)
        self.conv2 = GCNConv(dim_h, dim_h)
        self.conv3 = GCNConv(dim_h, dim_h)
        self.lin = Linear(dim_h, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        # Node embeddings 
        h = self.conv1(x, edge_index)
        h = h.relu()
        h = self.conv2(h, edge_index)
        h = h.relu()
        h = self.conv3(h, edge_index)

        # Graph-level readout
        hG = global_mean_pool(h, batch)

        # Classifier
        h = F.dropout(hG, p=0.5, training=self.training)
        h = self.lin(h)
        
        return hG, F.log_softmax(h, dim=1)

class GIN(torch.nn.Module):
    """GIN"""
    def __init__(self, dim_h):
        super(GIN, self).__init__()
        self.conv1 = GINConv(
            Sequential(Linear(dataset.num_node_features, dim_h),
                       BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv2 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv3 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.lin1 = Linear(dim_h*3, dim_h*3)
        self.lin2 = Linear(dim_h*3, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        # Node embeddings 
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)

        # Graph-level readout
        h1 = global_add_pool(h1, batch)
        h2 = global_add_pool(h2, batch)
        h3 = global_add_pool(h3, batch)

        # Concatenate graph embeddings
        h = torch.cat((h1, h2, h3), dim=1)

        # Classifie
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)
        
        return h, F.log_softmax(h, dim=1)

gcn = GCN(dim_h=32)
gin = GIN(dim_h=32)


def train(model, loader):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                      lr=0.01,
                                      weight_decay=0.01)
    epochs = 100

    model.train()
    for epoch in range(epochs+1):
        total_loss = 0
        acc = 0
        val_loss = 0
        val_acc = 0

        # Train on batches
        for data in loader:
          optimizer.zero_grad()
          _, out = model(data.x, data.edge_index, data.batch)
          loss = criterion(out, data.y)
          total_loss += loss / len(loader)
          acc += accuracy(out.argmax(dim=1), data.y) / len(loader)
          loss.backward()
          optimizer.step()

          # Validation
          val_loss, val_acc = test(model, val_loader)

    # Print metrics every 10 epochs
    if(epoch % 10 == 0):
        print(f'Epoch {epoch:>3} | Train Loss: {total_loss:.2f} '
              f'| Train Acc: {acc*100:>5.2f}% '
              f'| Val Loss: {val_loss:.2f} '
              f'| Val Acc: {val_acc*100:.2f}%')
          
    test_loss, test_acc = test(model, test_loader)
    print(f'Test Loss: {test_loss:.2f} | Test Acc: {test_acc*100:.2f}%')
    
    return model

@torch.no_grad()
def test(model, loader):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    loss = 0
    acc = 0

    for data in loader:
        _, out = model(data.x, data.edge_index, data.batch)
        loss += criterion(out, data.y) / len(loader)
        acc += accuracy(out.argmax(dim=1), data.y) / len(loader)

    return loss, acc

def accuracy(pred_y, y):
    """Calculate accuracy."""
    return ((pred_y == y).sum() / len(y)).item()

gcn = train(gcn, train_loader)
gin = train(gin, train_loader)