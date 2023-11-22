import numpy as np
import networkx as nx
from torch_geometric.utils import to_networkx
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

# グラフ構造の可視化関数(2D)
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


# グラフ構造の可視化関数(3D)
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

#サブグラフを見る
def check_subgraph(train_loader, val_loader, test_loader):
    print('\nTrain loader:')
    for i, subgraph in enumerate(train_loader):
        print(f' - Subgraph {i}: {subgraph}')

    print('\nValidation loader:')
    for i, subgraph in enumerate(val_loader):
        print(f' - Subgraph {i}: {subgraph}')

    print('\nTest loader:')
    for i, subgraph in enumerate(test_loader):
        print(f' - Subgraph {i}: {subgraph}')

#学習を可視化
def show_train(loss_list_train, loss_list_val):
    # 訓練回数のリストを生成（1から始まると仮定）
    epochs = range(100)



    # 訓練損失と検証損失をプロット
    loss_list_train_numpy = [loss.detach().numpy() for loss in loss_list_train]
    loss_list_test_numpy = [loss.detach().numpy() for loss in loss_list_val]

    # その後、通常通りプロットを行います。
    plt.plot(epochs, loss_list_train_numpy, label='Training loss')
    plt.plot(epochs, loss_list_test_numpy, label='Validation loss')

    # タイトルと軸ラベルを追加
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # 凡例を追加
    plt.legend()

    # グラフを表示
    plt.show()