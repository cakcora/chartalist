import os
import shutil
from collections import defaultdict
from multiprocessing import Process
import networkx as nx
import pandas as pd
import datetime as dt
from torch_geometric.data import Data
from typing import Any, Iterable, List, Optional, Tuple, Union
import torch
from torch import Tensor
import pickle

'''
    The following script will parse each transaction network and provide detailed stats about them. 

'''


class NetworkParser:
    # Path of the dataset folder
    file_path = "../data/all_network/"
    timeWindow = [7]
    # Validation duration condition
    networkValidationDuration = 20
    finalDataDuration = 5
    labelTreshholdPercentage = 10
    # Retrieve dataset by call to dataloader
    # Ethereum Stable Coin ERC20
    stat_data = pd.DataFrame(columns=['network', 'timeframe', 'start_date', 'num_nodes',
                                      'num_edges', 'density', 'diameter',
                                      'avg_shortest_path_length', 'max_degree_centrality',
                                      'min_degree_centrality',
                                      'max_closeness_centrality', 'min_closeness_centrality',
                                      'max_betweenness_centrality',
                                      'min_betweenness_centrality',
                                      'assortativity', 'clique_number', 'motifs', "peak", "last_dates_trans",
                                      "label_factor_percentage",
                                      "label"])

    stat_date = pd.DataFrame(columns=['network', "data_duration"])
    processingIndx = 1

    def processGraph(self, file):
        # load each network file
        # Timer(2, functools.partial(self.exitfunc, file_path, file)).start()
        print("Processing {}".format(file))
        selectedNetwork = pd.read_csv((self.file_path + file), sep=' ', names=["from", "to", "date", "value"])
        selectedNetwork['date'] = pd.to_datetime(selectedNetwork['date'], unit='s').dt.date

        # generate the label for this network
        date_counts = selectedNetwork.groupby(['date']).count()
        peak_count = date_counts['value'].max()

        # Calculate the sum of the value column for the last two dates
        last_date_sum = date_counts['value'].tail(self.finalDataDuration).sum()
        if (last_date_sum / peak_count) * 100 > self.labelTreshholdPercentage:
            label = "live"
        else:
            label = 'dead'

        for timeFrame in self.timeWindow:
            print("\nProcessing Timeframe {} ".format(timeFrame))
            transactionGraph = nx.DiGraph()
            start_date = selectedNetwork['date'].min()
            last_date_of_data = selectedNetwork['date'].max()
            # check if the network has more than 20 days of data
            if ((last_date_of_data - start_date).days < self.networkValidationDuration):
                print(file + "Is not a valid network")
                shutil.move(self.file_path + file, self.file_path + "Invalid/" + file)
                continue

            # select only the rows that fall within the first 7 days
            end_date = start_date + dt.timedelta(days=timeFrame)
            selectedNetworkInTimeFrame = selectedNetwork[
                (selectedNetwork['date'] >= start_date) & (selectedNetwork['date'] < end_date)]

            # Populate graph with edges
            for item in selectedNetworkInTimeFrame.to_dict(orient="records"):
                transactionGraph.add_edge(item["from"], item["to"], value=item["value"])

            with open('NetworkxGraphs/' + file, 'wb') as f:
                pickle.dump(transactionGraph, f)

            num_nodes = len(transactionGraph.nodes())
            num_edges = len(transactionGraph.edges())
            density = nx.density(transactionGraph)
            transactionGraph = transactionGraph.to_undirected()
            if (nx.is_connected(transactionGraph)):
                diameter = nx.diameter(transactionGraph)
                avg_shortest_path_length = nx.average_shortest_path_length(transactionGraph)
                clique_number = nx.graph_clique_number(transactionGraph)
            else:
                Gcc = sorted(nx.connected_components(transactionGraph), key=len, reverse=True)
                biggestConnectedComponent = transactionGraph.subgraph(Gcc[0])
                diameter = nx.diameter(biggestConnectedComponent)
                avg_shortest_path_length = nx.average_shortest_path_length(biggestConnectedComponent)
                clique_number = nx.graph_clique_number(biggestConnectedComponent)

            # try:
            #     diameter = max([max(j.values()) for (i,j) in nx.shortest_path_length(transactionGraph)])
            # except Exception as e:
            #     diameter = "Error"
            #
            # try:
            #     avg_shortest_path_length = nx.average_shortest_path_length(transactionGraph)
            # except Exception as e:
            #     avg_shortest_path_length = "NC"

            max_degree_centrality = max(nx.degree_centrality(transactionGraph).values())
            min_degree_centrality = min(nx.degree_centrality(transactionGraph).values())
            max_closeness_centrality = max(nx.closeness_centrality(transactionGraph).values())
            min_closeness_centrality = min(nx.closeness_centrality(transactionGraph).values())
            max_betweenness_centrality = max(nx.betweenness_centrality(transactionGraph).values())
            min_betweenness_centrality = min(nx.betweenness_centrality(transactionGraph).values())
            # max_eigenvector_centrality = max(nx.eigenvector_centrality(transactionGraph).values())
            # min_eigenvector_centrality = min(nx.eigenvector_centrality(transactionGraph).values())
            # pagerank = nx.pagerank(transactionGraph)
            assortativity = nx.degree_assortativity_coefficient(transactionGraph)
            # try:
            #     clique_number = nx.graph_clique_number(transactionGraph)
            # except Exception as e:
            #     clique_number = "NC"
            motifs = ""
            stats = {'network': file, 'timeframe': timeFrame, 'start_date': start_date, 'num_nodes': num_nodes,
                     'num_edges': num_edges, 'density': density, 'diameter': diameter,
                     'avg_shortest_path_length': avg_shortest_path_length,
                     'max_degree_centrality': max_degree_centrality,
                     'min_degree_centrality': min_degree_centrality,
                     'max_closeness_centrality': max_closeness_centrality,
                     'min_closeness_centrality': min_closeness_centrality,
                     'max_betweenness_centrality': max_betweenness_centrality,
                     'min_betweenness_centrality': min_betweenness_centrality,
                     'assortativity': assortativity, 'clique_number': clique_number, 'motifs': motifs,
                     "peak": peak_count, "last_dates_trans": last_date_sum,
                     "label_factor_percentage": (last_date_sum / peak_count),
                     "label": label}

            stat_data = self.stat_data.iloc[0:0]
            stat_data = stat_data.append(stats, ignore_index=True)
            stat_data.to_csv('final_data.csv', mode='a', header=False)
            shutil.move(self.file_path + file, self.file_path + "Processed/" + file)
            self.processingIndx += 1
            # nx.draw(transactionGraph, node_size=10)
            # plt.savefig("images/" + file + "_" + str(timeFrame) + "_" + ".png")

            print("\nFinisheng processing {} \n".format(file + "   " + str(timeFrame)))

    def creatNetworkGraphs(self, file):
        print("Processing {}".format(file))
        allOneNodeFeatures = {}
        sizeOfNodeFeatures = 5
        featureNames = []
        selectedNetwork = pd.read_csv((self.file_path + file), sep=' ', names=["from", "to", "date", "value"])
        selectedNetwork['date'] = pd.to_datetime(selectedNetwork['date'], unit='s').dt.date

        # generate the label for this network
        date_counts = selectedNetwork.groupby(['date']).count()
        peak_count = date_counts['value'].max()

        # Calculate the sum of the value column for the last two dates
        last_date_sum = date_counts['value'].tail(self.finalDataDuration).sum()
        if (last_date_sum / peak_count) * 100 > self.labelTreshholdPercentage:
            # live
            label = 1
        else:
            # dead
            label = 0

        for timeFrame in self.timeWindow:
            print("\nProcessing Timeframe {} ".format(timeFrame))
            transactionGraph = nx.DiGraph()
            start_date = selectedNetwork['date'].min()
            last_date_of_data = selectedNetwork['date'].max()
            # check if the network has more than 20 days of data
            if ((last_date_of_data - start_date).days < self.networkValidationDuration):
                print(file + "Is not a valid network")
                shutil.move(self.file_path + file, self.file_path + "Invalid/" + file)
                continue

            # select only the rows that fall within the first timeframe days
            end_date = start_date + dt.timedelta(days=timeFrame)
            selectedNetworkInTimeFrame = selectedNetwork[
                (selectedNetwork['date'] >= start_date) & (selectedNetwork['date'] < end_date)]

            for i in range(sizeOfNodeFeatures):
                allOneNodeFeatures["feature{}".format(i)] = 1.0
                featureNames.append("feature{}".format(i))

            # Populate graph with edges
            for item in selectedNetworkInTimeFrame.to_dict(orient="records"):
                transactionGraph.add_nodes_from([(item["from"], allOneNodeFeatures)])
                transactionGraph.add_nodes_from([(item["to"], allOneNodeFeatures)])
                transactionGraph.add_edge(item["from"], item["to"], value=item["value"])

            pygData = self.from_networkx(transactionGraph, label=label, group_node_attrs=featureNames)
            with open('PygGraphs/' + file, 'wb') as f:
                pickle.dump(pygData, f)

    def processDataDUration(self, file):
        # load each network file
        # Timer(2, functools.partial(self.exitfunc, file_path, file)).start()
        print("Processing {}".format(file))
        selectedNetwork = pd.read_csv((self.file_path + file), sep=' ', names=["from", "to", "date", "value"])
        selectedNetwork['date'] = pd.to_datetime(selectedNetwork['date'], unit='s').dt.date

        start_date = selectedNetwork['date'].min()
        last_date_of_data = selectedNetwork['date'].max()
        days = (last_date_of_data - start_date).days

        stats = {'network': file, "data_duration": days}

        stat_date = self.stat_date.iloc[0:0]
        stat_date = stat_date.append(stats, ignore_index=True)
        stat_date.to_csv('final_data_date.csv', mode='a', header=False)
        shutil.move(self.file_path + file, self.file_path + "Pr/" + file)

    def processMotifs(self, file):
        # Use Igraph to create the motifs
        print("Processing {}".format(file))
        selectedNetwork = pd.read_csv((self.file_path + file), sep=' ', names=["from", "to", "date", "value"])
        selectedNetwork['date'] = pd.to_datetime(selectedNetwork['date'], unit='s').dt.date

        for timeFrame in self.timeWindow:
            print("\nProcessing Timeframe {} ".format(timeFrame))
            transactionGraphs = nx.DiGraph()
            start_date = selectedNetwork['date'].min()
            last_date_of_data = selectedNetwork['date'].max()
            # check if the network has more than 20 days of data
            if ((last_date_of_data - start_date).days < self.networkValidationDuration):
                print(file + "Is not a valid network")
                shutil.move(self.file_path + file, self.file_path + "Invalid/" + file)
                continue

            # load each network file
            # Timer(2, functools.partial(self.exitfunc, file_path, file)).start()
            print("\nProcessing Timeframe {} ".format(timeFrame))
            transactionGraphs = nx.DiGraph()
            start_date = selectedNetwork['date'].min()
            last_date_of_data = selectedNetwork['date'].max()
            # check if the network has more than 20 days of data
            if ((last_date_of_data - start_date).days < self.networkValidationDuration):
                print(file + "Is not a valid network")
                shutil.move(self.file_path + file, self.file_path + "Invalid/" + file)
                continue

            # select only the rows that fall within the first 7 days
            end_date = start_date + dt.timedelta(days=timeFrame)
            selectedNetworkInTimeFrame = selectedNetwork[
                (selectedNetwork['date'] >= start_date) & (selectedNetwork['date'] < end_date)]

            # Populate graph with edges
            for item in selectedNetworkInTimeFrame.to_dict(orient="records"):
                transactionGraphs.add_edge(item["from"], item["to"], value=item["value"])

            motifs = None
            stats = {'network': file, 'motifs': motifs}

            stat_data = self.stat_data.iloc[0:0]
            stat_data = stat_data.append(stats, ignore_index=True)
            stat_data.to_csv('final_data_motifs.csv', mode='a', header=False)
            shutil.move(self.file_path + file, self.file_path + "Processed/" + file)
            self.processingIndx += 1
            # nx.draw(transactionGraphs, node_size=10)
            # plt.savefig("images/" + file + "_" + str(timeFrame) + "_" + ".png")

            print("\nFinisheng processing {} \n".format(file + "   " + str(timeFrame)))

    def main(self):
        print("Process Started\n")
        self.processingIndx = 0
        files = os.listdir(self.file_path)
        for file in files:
            if file.endswith(".txt"):
                print("Processing {} / {} \n".format(self.processingIndx, len(files) - 3))
                p = Process(target=self.creatNetworkGraphs, args=(file,))  # make process
                p.start()  # start function
                p.join(timeout=180)

                # Check if the process is still running
                if p.is_alive():
                    # The process is still running, terminate it
                    p.terminate()
                    print("The file is taking infinite time - check the file ")
                    shutil.move(self.file_path + file, self.file_path + "issue/" + file)
                    self.processingIndx += 1
                    print("Function timed out and was terminated")
                else:
                    # The process has finished
                    self.processingIndx += 1
                    print("Process finished successfully")
        # stat_data.to_csv("final_data.csv")

    def from_networkx(
            self, G: Any, label: int,
            group_node_attrs: Optional[Union[List[str], all]] = None,
            group_edge_attrs: Optional[Union[List[str], all]] = None,
    ) -> 'torch_geometric.data.Data':
        r"""Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
        :class:`torch_geometric.data.Data` instance.

        Args:
            G (networkx.Graph or networkx.DiGraph): A networkx graph.
            group_node_attrs (List[str] or all, optional): The node attributes to
                be concatenated and added to :obj:`data.x`. (default: :obj:`None`)
            group_edge_attrs (List[str] or all, optional): The edge attributes to
                be concatenated and added to :obj:`data.edge_attr`.
                (default: :obj:`None`)

        .. note::

            All :attr:`group_node_attrs` and :attr:`group_edge_attrs` values must
            be numeric.

        Examples:

            # >>> edge_index = torch.tensor([
            # ...     [0, 1, 1, 2, 2, 3],
            # ...     [1, 0, 2, 1, 3, 2],
            # ... ])
            # >>> data = Data(edge_index=edge_index, num_nodes=4)
            # >>> g = to_networkx(data)
            # >>> # A `Data` object is returned
            # >>> from_networkx(g)
            Data(edge_index=[2, 6], num_nodes=4)
        """
        import networkx as nx

        from torch_geometric.data import Data

        G = nx.convert_node_labels_to_integers(G)
        G = G.to_directed() if not nx.is_directed(G) else G

        if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
            edges = list(G.edges(keys=False))
        else:
            edges = list(G.edges)

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        data = defaultdict(list)

        if G.number_of_nodes() > 0:
            node_attrs = list(next(iter(G.nodes(data=True)))[-1].keys())
        else:
            node_attrs = {}

        if G.number_of_edges() > 0:
            edge_attrs = list(next(iter(G.edges(data=True)))[-1].keys())
        else:
            edge_attrs = {}

        for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
            if set(feat_dict.keys()) != set(node_attrs):
                raise ValueError('Not all nodes contain the same attributes')
            for key, value in feat_dict.items():
                data[str(key)].append(value)

        for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
            if set(feat_dict.keys()) != set(edge_attrs):
                raise ValueError('Not all edges contain the same attributes')
            for key, value in feat_dict.items():
                key = f'edge_{key}' if key in node_attrs else key
                data[str(key)].append(value)

        for key, value in G.graph.items():
            key = f'graph_{key}' if key in node_attrs else key
            data[str(key)] = value

        for key, value in data.items():
            if isinstance(value, (tuple, list)) and isinstance(value[0], Tensor):
                data[key] = torch.stack(value, dim=0)
            else:
                try:
                    data[key] = torch.tensor(value)
                except (ValueError, TypeError):
                    pass

        data['edge_index'] = edge_index.view(2, -1)
        data = Data.from_dict(data)
        data['y'] = label

        if group_node_attrs is all:
            group_node_attrs = list(node_attrs)
        if group_node_attrs is not None:
            xs = []
            for key in group_node_attrs:
                x = data[key]
                x = x.view(-1, 1) if x.dim() <= 1 else x
                xs.append(x)
                del data[key]
            data.x = torch.cat(xs, dim=-1)

        if group_edge_attrs is all:
            group_edge_attrs = list(edge_attrs)
        if group_edge_attrs is not None:
            xs = []
            for key in group_edge_attrs:
                key = f'edge_{key}' if key in node_attrs else key
                x = data[key]
                x = x.view(-1, 1) if x.dim() <= 1 else x
                xs.append(x)
                del data[key]
            data.edge_attr = torch.cat(xs, dim=-1)

        if data.x is None and data.pos is None:
            data.num_nodes = G.number_of_nodes()

        return data


if __name__ == '__main__':
    np = NetworkParser()
    np.main()
