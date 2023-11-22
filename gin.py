from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

from utils import check_graph, check_graphs, draw_graph, draw_graph_3d, check_subgraph
from models import GCN, GIN
from train import train

def main():
    dataset = TUDataset(root='.', name='PROTEINS').shuffle()
    #check_graphs(dataset)
    #check_graph(dataset[0])
    #draw_graph(dataset[0])
    #draw_graph_3d(dataset[0])
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

    #check_subgraph(train_loader, val_loader, test_loader)

    gcn = GCN(32, dataset.num_node_features, dataset.num_classes)
    gin = GIN(32, dataset.num_node_features, dataset.num_classes)
    gcn = train(gcn, train_loader, val_loader, test_loader)
    gin = train(gin, train_loader, val_loader, test_loader)

if __name__ == "__main__":
    main()