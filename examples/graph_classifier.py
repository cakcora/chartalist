import os
import pickle
import random

import pandas as pd
import yaml
from matplotlib import pyplot as plt
from pycaret.classification import *
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader
import torch
from torch.nn import Sequential, Linear, ReLU
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool, GINConv, global_mean_pool
import pytorch_lightning as pl
from torch.nn import BatchNorm1d

random.seed(123)


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_node_features, num_classes):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        # TODO list of conv layers
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        # TODO add dropout here
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


class GIN(torch.nn.Module):

    def __init__(self, dim_features, dim_target, config):
        super(GIN, self).__init__()

        self.config = config
        self.dropout = config['dropout'][0]
        self.embeddings_dim = [config['hidden_units'][0][0]] + config['hidden_units'][0]
        self.no_layers = len(self.embeddings_dim)
        self.first_h = []
        self.nns = []
        self.convs = []
        self.linears = []

        train_eps = config['train_eps'][0]
        if config['aggregation'][0] == 'sum':
            self.pooling = global_add_pool
        elif config['aggregation'][0] == 'mean':
            self.pooling = global_mean_pool

        for layer, out_emb_dim in enumerate(self.embeddings_dim):

            if layer == 0:
                self.first_h = Sequential(Linear(dim_features, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU(),
                                          Linear(out_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU())
                self.linears.append(Linear(out_emb_dim, dim_target))
            else:
                input_emb_dim = self.embeddings_dim[layer - 1]
                self.nns.append(Sequential(Linear(input_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU(),
                                           Linear(out_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU()))
                self.convs.append(GINConv(self.nns[-1], train_eps=train_eps))  # Eq. 4.2

                self.linears.append(Linear(out_emb_dim, dim_target))

        self.nns = torch.nn.ModuleList(self.nns)
        self.convs = torch.nn.ModuleList(self.convs)
        self.linears = torch.nn.ModuleList(self.linears)  # has got one more for initial input

    def forward(self, x, edge_index, batch):
        out = 0

        for layer in range(self.no_layers):
            if layer == 0:
                x = self.first_h(x)
                out += F.dropout(self.pooling(self.linears[layer](x), batch), p=self.dropout)
            else:
                # Layer l ("convolution" layer)
                x = self.convs[layer - 1](x, edge_index)
                out += F.dropout(self.linears[layer](self.pooling(x, batch)), p=self.dropout, training=self.training)

        return out


def read_data():
    data = pd.read_csv("final_data" + ".csv", header=0)
    date_data = pd.read_csv("GnnResults/final_data_date.csv", header=0)
    avg_trans_data = pd.read_csv("GnnResults/average_transaction.csv", header=0)
    date_data.columns = ['id', 'network', 'data_duration']
    avg_trans_data.columns = ['id', 'network', 'timeframe', 'avg_daily_trans']
    data.columns = ['id', 'network', 'timeframe', 'start_date', 'node_count',
                    'edge_count', 'density', 'diameter',
                    'avg_shortest_path_length', 'max_degree_centrality',
                    'min_degree_centrality',
                    'max_closeness_centrality', 'min_closeness_centrality',
                    'max_betweenness_centrality',
                    'min_betweenness_centrality',
                    'assortativity', 'clique_number', 'motifs', "peak", "last_dates_trans",
                    "label_factor_percentage",
                    "label"]

    data = data.drop('id', axis=1)
    date_data = date_data.drop('id', axis=1)
    avg_trans_data = avg_trans_data.drop('id', axis=1)
    avg_trans_data = avg_trans_data.drop('timeframe', axis=1)
    data = pd.merge(data, date_data, on="network", how="left")
    data = pd.merge(data, avg_trans_data, on="network", how="left")
    data = data.drop_duplicates(subset=["network"], keep='first')
    data['label'] = data.pop('label')
    data.to_csv('final_data_with_header.csv', header=True)
    return data


def read_torch_data():
    file_path = "PygGraphs/"
    inx = 1
    GraphDataList = []
    files = os.listdir(file_path)
    for file in files:
        if file.endswith(".txt"):
            with open(file_path + file, 'rb') as f:
                print("\n Reading Torch Data {} / {}".format(inx, len(files)))
                data = pickle.load(f)
                GraphDataList.append(data)
                inx += 1
    return GraphDataList


def read_torch_time_series_data(network, variable= None):
    file_path = "PygGraphs/TimeSeries/{}/".format(network)
    file_path_TDA = "PygGraphs/TimeSeries/{}/TDA/".format(network)
    file_path_different_TDA = "PygGraphs/TimeSeries/{}/TDA/{}/".format(network, variable)
    inx = 1
    GraphDataList = []
    files = os.listdir(file_path_different_TDA)
    for file in files:
        with open(file_path_different_TDA + file, 'rb') as f:
            # print("\n Reading Torch Data {} / {}".format(inx, len(files)))
            data = pickle.load(f)
            GraphDataList.append(data)
            inx += 1
    return GraphDataList


def data_by_edge_visualization(data):
    # Divide node_count into 10 equally sized buckets
    bins = [0, 5, 10, 50, 100, 500, 1000, 2000, 5000, 10000]
    labels = ["0 - 5", "5 - 10", "10 - 50", "50 - 100", "100 - 500", "500 - 1K", "1K - 2K", "2K - 5K", "5K - inf"]
    data['bucket'] = pd.cut(data['edge_count'], bins=bins, labels=labels)

    # Group by bucket and label and count the number of 0's and 1's in each group
    grouped = data.groupby(['bucket', 'label'])['edge_count'].count().unstack()

    # Plot the results as a bar chart
    ax = grouped.plot(kind='bar')
    ax.set_xlabel('Unique Edge Count')
    ax.xaxis.set_label_position('top')
    ax.set_ylabel('Network Count')
    ax.legend(title='Label', loc='upper right')
    plt.xticks(rotation=45)
    plt.savefig('Edge.png', dpi=300)
    plt.show()


def data_by_node_visualization(data):
    # Divide node_count into 10 equally sized buckets
    bins = [0, 5, 10, 50, 100, 500, 1000, 2000, 5000, 10000]
    labels = ["0 - 5", "5 - 10", "10 - 50", "50 - 100", "100 - 500", "500 - 1K", "1K - 2K", "2K - 5K", "5K - inf"]
    data['bucket'] = pd.cut(data['node_count'], bins=bins, labels=labels)

    # Group by bucket and label and count the number of 0's and 1's in each group
    grouped = data.groupby(['bucket', 'label'])['node_count'].count().unstack()

    # Plot the results as a bar chart
    ax = grouped.plot(kind='bar')
    ax.set_xlabel('Node Count')
    ax.xaxis.set_label_position('top')
    ax.set_ylabel('Network Count')
    ax.legend(title='Label', loc='upper right')
    plt.xticks(rotation=45)
    plt.savefig('Node.png', dpi=300)
    plt.show()


def data_by_density_visualization(data):
    # Divide node_count into 10 equally sized buckets
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    labels = ["0 - 0.1", "0.1 - 0.2", "0.2 - 0.3", "0.3 - 0.4", "0.4 - 0.5", "0.5 - 0.6", "0.6 - 0.7", "0.7 - 0.8",
              "0.8 - 0.9", "0.9 - 1"]
    data['bucket'] = pd.cut(data['density'], bins=bins, labels=labels)

    # Group by bucket and label and count the number of 0's and 1's in each group
    grouped = data.groupby(['bucket', 'label'])['density'].count().unstack()

    # Plot the results as a bar chart
    ax = grouped.plot(kind='bar')
    ax.xaxis.set_label_position('top')
    ax.set_xlabel('Density')
    ax.set_ylabel('Network Count')
    ax.legend(title='Label', loc='upper right')
    plt.xticks(rotation=45)
    plt.savefig('Density.png', dpi=300)
    plt.show()


def data_by_peak_visualization(data):
    # Divide node_count into 10 equally sized buckets
    bins = [0, 5, 10, 50, 100, 500, 1000, 2000, 5000, 10000]
    labels = ["0 - 5", "5 - 10", "10 - 50", "50 - 100", "100 - 500", "500 - 1K", "1K - 2K", "2K - 5K", "5K - inf"]
    data['bucket'] = pd.cut(data['peak'], bins=bins, labels=labels)

    # Group by bucket and label and count the number of 0's and 1's in each group
    grouped = data.groupby(['bucket', 'label'])['peak'].count().unstack()

    # Plot the results as a bar chart
    ax = grouped.plot(kind='bar')
    ax.xaxis.set_label_position('top')
    ax.set_xlabel('Number of Peak Transactions')
    ax.set_ylabel('Network Count')
    ax.legend(title='Label', loc='upper right')
    plt.xticks(rotation=45)
    plt.savefig('Peak.png', dpi=300)
    plt.show()


def data_by_data_duration_visualization(data):
    # Divide node_count into 10 equally sized buckets
    bins = [20, 50, 100, 150, 200, 250, 300, 350, 400]
    labels = ["20 - 50", "50 - 100", "100 - 150", "150 - 200", "200 - 250", "250 - 300", "300 - 350", "350 - inf"]
    data['bucket'] = pd.cut(data['data_duration'], bins=bins, labels=labels)

    # Group by bucket and label and count the number of 0's and 1's in each group
    grouped = data.groupby(['bucket', 'label'])['data_duration'].count().unstack()

    # Plot the results as a bar chart
    ax = grouped.plot(kind='bar')
    ax.xaxis.set_label_position('top')
    ax.set_xlabel('Days of having active transactions')
    ax.set_ylabel('Network Count')
    plt.xticks(rotation=45)
    ax.legend(title='Label', loc='upper right')
    plt.savefig('Duration.png', dpi=300)
    plt.show()


def data_by_Avg_shortest_path_length_visualization(data):
    # Divide node_count into 10 equally sized buckets
    bins = [0, 2, 5, 10, 15, 20]
    labels = ["0 - 2", "2 - 5", "5 - 10", "10 - 15", "15 - inf"]
    data['bucket'] = pd.cut(data['avg_shortest_path_length'], bins=bins, labels=labels)

    # Group by bucket and label and count the number of 0's and 1's in each group
    grouped = data.groupby(['bucket', 'label'])['avg_shortest_path_length'].count().unstack()

    # Plot the results as a bar chart
    ax = grouped.plot(kind='bar')
    ax.xaxis.set_label_position('top')
    ax.set_xlabel('Avg Shortest Path Length')
    ax.set_ylabel('Network Count')
    plt.xticks(rotation=45)
    ax.legend(title='Label', loc='upper right')
    plt.savefig('Avg_shortest_path_length.png', dpi=300)
    plt.show()


def data_by_max_degree_centrality_visualization(data):
    # Divide node_count into 10 equally sized buckets
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    labels = ["0 - 0.1", "0.1 - 0.2", "0.2 - 0.3", "0.3 - 0.4", "0.4 - 0.5", "0.5 - 0.6", "0.6 - 0.7", "0.7 - 0.8",
              "0.8 - 0.9", "0.9 - 1"]
    data['bucket'] = pd.cut(data['max_degree_centrality'], bins=bins, labels=labels)

    # Group by bucket and label and count the number of 0's and 1's in each group
    grouped = data.groupby(['bucket', 'label'])['max_degree_centrality'].count().unstack()

    # Plot the results as a bar chart
    ax = grouped.plot(kind='bar')
    ax.xaxis.set_label_position('top')
    ax.set_xlabel('Max Degree Centrality')
    ax.set_ylabel('Network Count')
    ax.legend(title='Label', loc='upper left')
    plt.xticks(rotation=45)
    plt.savefig('max_degree_centrality.png', dpi=300)
    plt.show()


def data_by_max_closeness_centrality_visualization(data):
    # Divide node_count into 10 equally sized buckets
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    labels = ["0 - 0.1", "0.1 - 0.2", "0.2 - 0.3", "0.3 - 0.4", "0.4 - 0.5", "0.5 - 0.6", "0.6 - 0.7", "0.7 - 0.8",
              "0.8 - 0.9", "0.9 - 1"]
    data['bucket'] = pd.cut(data['max_closeness_centrality'], bins=bins, labels=labels)

    # Group by bucket and label and count the number of 0's and 1's in each group
    grouped = data.groupby(['bucket', 'label'])['max_closeness_centrality'].count().unstack()

    # Plot the results as a bar chart
    ax = grouped.plot(kind='bar')
    ax.set_xlabel('Max Closeness Centrality')
    ax.set_ylabel('Network Count')
    plt.xticks(rotation=45)
    ax.xaxis.set_label_position('top')
    ax.legend(title='Label', loc='upper left')
    plt.savefig('max_closeness_centrality.png', dpi=300)
    plt.show()


def data_by_max_betweenness_centrality_visualization(data):
    # Divide node_count into 10 equally sized buckets
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    labels = ["0 - 0.1", "0.1 - 0.2", "0.2 - 0.3", "0.3 - 0.4", "0.4 - 0.5", "0.5 - 0.6", "0.6 - 0.7", "0.7 - 0.8",
              "0.8 - 0.9", "0.9 - 1"]
    data['bucket'] = pd.cut(data['max_betweenness_centrality'], bins=bins, labels=labels)

    # Group by bucket and label and count the number of 0's and 1's in each group
    grouped = data.groupby(['bucket', 'label'])['max_betweenness_centrality'].count().unstack()

    # Plot the results as a bar chart

    ax = grouped.plot(kind='bar')
    ax.xaxis.set_label_position('top')
    ax.set_xlabel('Max Betweenness Centrality')
    ax.set_ylabel('Network Count')
    plt.xticks(rotation=45)
    ax.legend(title='Label', loc='upper left')
    plt.savefig('max_betweenness_centrality.png', dpi=300)
    plt.show()


def data_by_clique_visualization(data):
    # Divide node_count into 10 equally sized buckets
    bins = [0, 1, 2, 3, 4, 5, 6, 7]
    labels = ["1", "2", "3", "4", "5", "6 - inf"]
    data['bucket'] = pd.cut(data['clique_number'], bins=bins, labels=labels)

    # Group by bucket and label and count the number of 0's and 1's in each group
    grouped = data.groupby(['bucket', 'label'])['clique_number'].count().unstack()

    # Plot the results as a bar chart

    ax = grouped.plot(kind='bar')
    ax.xaxis.set_label_position('top')
    ax.set_xlabel('Clique Count')
    ax.set_ylabel('Network Count')
    plt.xticks(rotation=45)
    ax.legend(title='Label', loc='upper left')
    plt.savefig('clique_number.png', dpi=300)
    plt.show()


def data_by_avg_daily_trans_visualization(data):
    # Divide node_count into 10 equally sized buckets
    bins = [0, 5, 10, 20, 50, 100, 150, 200]
    labels = ["5", "10", "20", "50", "100", "150", "200 - inf"]
    data['bucket'] = pd.cut(data['avg_daily_trans'], bins=bins, labels=labels)

    # Group by bucket and label and count the number of 0's and 1's in each group
    grouped = data.groupby(['bucket', 'label'])['avg_daily_trans'].count().unstack()

    # Plot the results as a bar chart

    ax = grouped.plot(kind='bar')
    ax.xaxis.set_label_position('top')
    ax.set_xlabel('Average daily transactions Count')
    ax.set_ylabel('Network Count')
    plt.xticks(rotation=45)
    ax.legend(title='Label', loc='upper right')
    plt.savefig('avg_daily_trans_number.png', dpi=300)
    plt.show()


def classifier(data):
    # cleaning features
    data = data.drop('network', axis=1)
    data = data.drop('timeframe', axis=1)
    data = data.drop('start_date', axis=1)
    data = data.drop('label_factor_percentage', axis=1)
    data = data.drop('last_dates_trans', axis=1)
    data = data.drop('data_duration', axis=1)
    data = data.drop('peak', axis=1)

    # 791 dead, 824 live
    # des = data.groupby("label")

    train = data.sample(frac=0.7, random_state=200)
    test = data.drop(train.index)
    # Setting up the Environment
    clf = setup(data=train, target='label')

    # Comparing All Models
    best_model = compare_models()
    # evaluate_model(best_model)
    plot_model(best_model, plot='feature', save=True)
    predictions = predict_model(best_model, data=test)
    pred_acc = (pull().set_index('Model'))['Accuracy'].iloc[0]


def GCN_classifier(data):
    for duplication in range(0, 5):
        train_size = int(0.8 * len(data))
        test_size = len(data) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])
        train_loader = DataLoader(train_dataset, shuffle=True)
        test_loader = DataLoader(test_dataset)
        model = GCN(hidden_channels=64, num_node_features=5, num_classes=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(0, 101):
            train(train_loader, model, criterion, optimizer)
            scores_tr = test(train_loader, model)
            train_acc = scores_tr[0]
            train_auc = scores_tr[1]
            scores_te = test(test_loader, model)
            test_acc = scores_te[0]
            test_auc = scores_te[1]
            if epoch % 10 == 0:
                print(f"Duplicate\t{duplication},Epoch\t {epoch}\t "
                      f"Train Accuracy\t {train_acc:.4f}\t Train AUC Score: {train_auc:.4f}\t "
                      f"Test Accuracy: {test_acc:.4f}\t Test AUC Score: {test_auc:.4f}")


def GIN_classifier(data, network):
    count_one_labels = sum(1 for item in data if item['y'] == 1)
    count_zero_labels = sum(1 for item in data if item['y'] == 0)
    with open("config_GIN.yml", "r") as f:
        config = yaml.load(f)
    for duplication in range(0, 1):
        train_size = int(0.8 * len(data))
        test_size = len(data) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])
        train_loader = DataLoader(train_dataset, shuffle=True)
        test_loader = DataLoader(test_dataset)
        model = GIN(dim_features=1, dim_target=2, config=config)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(0, 101):
            train(train_loader, model, criterion, optimizer)
            scores_tr = test(train_loader, model)
            train_acc = scores_tr[0]
            train_auc = scores_tr[1]
            scores_te = test(test_loader, model)
            test_acc = scores_te[0]
            test_auc = scores_te[1]
            scores_unseen = test(test_loader, model)
            unseen_acc = scores_unseen[0]
            unseen_auc = scores_unseen[1]
            if epoch % 10 == 0:
                with open('GnnResults/GIN_TimeSeries_Result.txt', 'a+') as file:
                    file.write(
                        f"\nNetwork\t{network}\tDuplicate\t{duplication}\tEpoch\t{epoch}\tTrain Accuracy\t{train_acc:.4f}\tTrain AUC Score\t{train_auc:.4f}\tTest Accuracy:{test_acc:.4f}\tTest AUC Score\t{test_auc:.4f}\tunseen AUC Score\t{unseen_auc:.4f}\tNumber of Zero labels\t{count_zero_labels}\tNumber of one labels\t{count_one_labels}")
                    file.close()
                print(
                    f"Network\t{network} Duplicate\t{duplication}\tEpoch\t {epoch}\t Train Accuracy\t {train_acc:.4f}\t Train AUC Score\t {train_auc:.4f}\t Test Accuracy: {test_acc:.4f}\t test AUC Score\t {test_auc:.4f}\t unseen AUC Score\t {unseen_auc:.4f}")


def train(train_loader, model, criterion, optimizer):
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
        try:
            out = model(data.x.type(torch.float32), data.edge_index, data.batch)  # Perform a single forward pass.
            loss = criterion(out, data.y)  # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.
        except Exception as e:
            # print(str(e) + "In train")
            continue


def test(test_loader, model):
    model.eval()

    correct = 0
    auc_score = 0
    total_samples = 0

    y_true_list = []
    y_score_list = []

    for data in test_loader:  # Iterate in batches over the training/test dataset.
        try:
            out = model(data.x.type(torch.float32), data.edge_index, data.batch)
            pred = out.argmax(dim=1)  # Use the class with the highest probability.

            correct += int((pred == data.y).sum().item())  # Check against ground-truth labels.
            total_samples += data.y.size(0)

            arr2 = out[:, 1].detach().numpy()
            y_score_list.append(arr2[0])
            arr1 = data.y.detach().numpy()
            y_true_list.append(arr1[0])
        except Exception as e:
            # print(str(e) + "In Test")
            continue

    try:
        auc_score += roc_auc_score(y_true=y_true_list, y_score=y_score_list, multi_class='ovr', average='weighted')
    except Exception as e:
        print(e)
        pass

    accuracy = correct / total_samples
    # auc_score /= len(test_loader)

    # print(f"Accuracy: {accuracy:.4f}, AUC Score: {auc_score:.4f}")

    return accuracy, auc_score


# def GNN_classifier(data):


if __name__ == "__main__":
    networkList = ["networkadex.txt"]
    tdaDifferentGraph = ["Overlap_0.1_Ncube_2", "Overlap_0.1_Ncube_5", "Overlap_0.2_Ncube_2", "Overlap_0.2_Ncube_5", "Overlap_0.3_Ncube_2", "Overlap_0.3_Ncube_5", "Overlap_0.5_Ncube_2", "Overlap_0.5_Ncube_5", "Overlap_0.6_Ncube_2", "Overlap_0.6_Ncube_5"]
    # data = read_data()
    # data_by_edge_visualization(data)
    # data_by_node_visualization(data)
    # data_by_density_visualization(data)
    # data_by_peak_visualization(data)
    # data_by_data_duration_visualization(data)
    # data_by_Avg_shortest_path_length_visualization(data)
    # data_by_max_degree_centrality_visualization(data)
    # data_by_max_closeness_centrality_visualization(data)
    # data_by_max_betweenness_centrality_visualization(data)
    # data_by_clique_visualization(data)
    # data_by_avg_daily_trans_visualization(data)
    # classifier(data)
    for network in networkList:
        for tdaVariable in tdaDifferentGraph:
            print("Working on {} in {} \n".format(network, tdaVariable))
            data = read_torch_time_series_data(network, tdaVariable)
            GIN_classifier(data, network)
    # GCN_classifier(data)
