import random
from matplotlib import pyplot as plt
from pycaret.classification import *

random.seed(123)


def read_data():
    data = pd.read_csv("final_data" + ".csv", header=0)
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
    data.to_csv('final_data_with_header.csv', header=True)
    return data


def data_by_edge_visualization(data):
    # Divide node_count into 10 equally sized buckets
    bins = [0, 5, 10, 50, 100, 500, 1000, 2000, 5000, 10000]
    data['bucket'] = pd.cut(data['edge_count'], bins=bins, labels=bins[1:])

    # Group by bucket and label and count the number of 0's and 1's in each group
    grouped = data.groupby(['bucket', 'label'])['edge_count'].count().unstack()

    # Plot the results as a bar chart
    ax = grouped.plot(kind='bar', stacked=True, figsize=(10, 9))
    ax.set_xlabel('Edge Count')
    ax.set_ylabel('#')
    ax.legend(title='Label', loc='upper right')
    plt.savefig('Edge.png')
    plt.show()


def data_by_node_visualization(data):
    # Divide node_count into 10 equally sized buckets
    bins = [0, 5, 10, 50, 100, 500, 1000, 2000, 5000, 10000]
    data['bucket'] = pd.cut(data['node_count'], bins=bins, labels=bins[1:])

    # Group by bucket and label and count the number of 0's and 1's in each group
    grouped = data.groupby(['bucket', 'label'])['node_count'].count().unstack()

    # Plot the results as a bar chart
    ax = grouped.plot(kind='bar', stacked=True, figsize=(10, 9))
    ax.set_xlabel('Node Count')
    ax.set_ylabel('#')
    ax.legend(title='Label', loc='upper right')
    plt.savefig('Node.png')
    plt.show()


def classifier(data):
    data = data.drop('network', axis=1)
    data = data.drop('timeframe', axis=1)
    data = data.drop('start_date', axis=1)
    train = data.sample(frac=0.7, random_state=200)
    test = data.drop(train.index)
    # Setting up the Environment
    clf = setup(data=train, target='label')

    # Comparing All Models
    best_model = compare_models()
    # evaluate_model(best_model)
    predictions = predict_model(best_model, data=test)
    pred_acc = (pull().set_index('Model'))['Accuracy'].iloc[0]


if __name__ == "__main__":
    data = read_data()
    data_by_edge_visualization(data)
    data_by_node_visualization(data)
    classifier(data)
