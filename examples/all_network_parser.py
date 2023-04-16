import ast
import os
import shutil
from collections import defaultdict
from multiprocessing import Process
import networkx as nx
import pandas as pd
import datetime as dt
import numpy as np
import multiprocessing as mp
import numpy as nump

from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from numpy import long
from torch_geometric.data import Data
from typing import Any, Iterable, List, Optional, Tuple, Union
import torch
from torch import Tensor
import pickle
import matplotlib.ticker as mticker
import json

'''
    The following script will parse each transaction network and provide detailed stats about them. 

'''


class NetworkParser:
    # Path of the dataset folder
    file_path = "../data/all_network/"
    timeseries_file_path = "../data/all_network/TimeSeries/"
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

    def creatGraphFeatures(self, file):
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
        selectedNetwork['value'] = selectedNetwork['value'].astype(float)

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

        # normalize the edge weights for the graph network {0-9}
        max_transfer = float(selectedNetwork['value'].max())
        min_transfer = float(selectedNetwork['value'].min())

        selectedNetwork['value'] = selectedNetwork['value'].apply(
            lambda x: 1 + (9 * ((float(x) - min_transfer) / (max_transfer - min_transfer))))

        for timeFrame in self.timeWindow:
            print("\nProcessing Timeframe {} ".format(timeFrame))
            transactionGraph = nx.MultiDiGraph()
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

            # for i in range(sizeOfNodeFeatures):
            #     allOneNodeFeatures["feature{}".format(i)] = 1.0
            #     featureNames.append("feature{}".format(i))

            # group by for extracting node features
            outgoing_weight_sum = (selectedNetwork.groupby(by=['from'])['value'].sum())
            incoming_weight_sum = (selectedNetwork.groupby(by=['to'])['value'].sum())
            outgoing_count = (selectedNetwork.groupby(by=['from'])['value'].count())
            incoming_count = (selectedNetwork.groupby(by=['to'])['value'].count())

            # Populate graph with edges
            for item in selectedNetworkInTimeFrame.to_dict(orient="records"):
                from_node_features = {}
                to_node_features = {}
                # calculating node features for each edge
                # feature 1 -> sum of outgoing edge weights
                from_node_features["outgoing_edge_weight_sum"] = outgoing_weight_sum[item['from']]

                try:
                    to_node_features["outgoing_edge_weight_sum"] = outgoing_weight_sum[item['to']]
                except Exception as e:
                    to_node_features["outgoing_edge_weight_sum"] = 0

                # feature 2 -> sum of incoming edge weights
                to_node_features["incoming_edge_weight_sum"] = incoming_weight_sum[item['to']]
                try:
                    from_node_features["incoming_edge_weight_sum"] = incoming_weight_sum[item['from']]
                except Exception as e:
                    from_node_features["incoming_edge_weight_sum"] = 0
                # feature 3 -> number of outgoing edges
                from_node_features["outgoing_edge_count"] = outgoing_count[item['from']]
                try:
                    to_node_features["outgoing_edge_count"] = outgoing_count[item['to']]
                except Exception as e:
                    to_node_features["outgoing_edge_count"] = 0

                # feature 4 -> number of incoming edges
                to_node_features["incoming_edge_count"] = incoming_count[item['to']]
                try:
                    from_node_features["incoming_edge_count"] = incoming_count[item['from']]
                except Exception as e:
                    from_node_features["incoming_edge_count"] = 0

                # # feature 5 -> max outgoing edge weight
                # from_node_features["outgoing_edge_weight_max"] = selectedNetwork.groupby(by=['from'])['value'].max
                # # feature 6 -> max incoming edge weight
                # from_node_features["incoming_edge_weight_max"] = selectedNetwork.groupby(by=['from'])['value'].max
                # to_node_features["incoming_edge_weight_max"] = selectedNetwork.groupby(by=['from'])['value'].max
                # # feature 7 -> min outgoing edge weight
                # from_node_features["outgoing_edge_weight_min"] = selectedNetwork.groupby(by=['from'])['value'].min
                # to_node_features["outgoing_edge_weight_min"] = selectedNetwork.groupby(by=['from'])['value'].min
                # # feature 8 -> min incoming edge weight
                # from_node_features["incoming_edge_weight_min"] = selectedNetwork.groupby(by=['from'])['value'].min
                # to_node_features["incoming_edge_weight_min"] = selectedNetwork.groupby(by=['from'])['value'].min

                transactionGraph.add_nodes_from([(item["from"], from_node_features)])
                transactionGraph.add_nodes_from([(item["to"], to_node_features)])
                transactionGraph.add_edge(item["from"], item["to"], value=item["value"])
            featureNames = ["outgoing_edge_weight_sum", "incoming_edge_weight_sum", "outgoing_edge_count",
                            "incoming_edge_count"]
            pygData = self.from_networkx(transactionGraph, label=label, group_node_attrs=featureNames)
            with open('PygGraphs/' + file, 'wb') as f:
                pickle.dump(pygData, f)


    def creatTimeSeriesGraphs(self, file):
        print("Processing {}".format(file))
        windowSize = 7  # Day
        gap = 3
        lableWindowSize = 7  # Day
        maxDuration = 180  # Day
        indx = 0

        selectedNetwork = pd.read_csv((self.timeseries_file_path + file), sep=' ', names=["from", "to", "date", "value"])
        selectedNetwork['date'] = pd.to_datetime(selectedNetwork['date'], unit='s').dt.date
        selectedNetwork['value'] = selectedNetwork['value'].astype(float)
        selectedNetwork = selectedNetwork.sort_values(by='date')
        window_start_date = selectedNetwork['date'].min()
        data_last_date = selectedNetwork['date'].max()

        #print("\n {} Days OF Data -> {} ".format(file, (data_last_date - window_start_date).days ))
        # check if the network has more than 20 days of data
        if ((data_last_date - window_start_date).days < maxDuration):
            print(file + "Is not a valid network")
            shutil.move(self.file_path + file, self.file_path + "Invalid/" + file)
            return

        # normalize the edge weights for the graph network {0-9}
        max_transfer = float(selectedNetwork['value'].max())
        min_transfer = float(selectedNetwork['value'].min())
        selectedNetwork['value'] = selectedNetwork['value'].apply(
            lambda x: 1 + (9 * ((float(x) - min_transfer) / (max_transfer - min_transfer))))

        # Graph Generation Process and Labeling
        while (data_last_date - window_start_date).days > (windowSize + gap + lableWindowSize):
            print("\nRemaining Process {} ".format(
                (data_last_date - window_start_date).days / (windowSize + gap + lableWindowSize)))
            indx += 1
            transactionGraph = nx.MultiDiGraph()


            # select window data
            window_end_date = window_start_date + dt.timedelta(days=windowSize)
            selectedNetworkInGraphDataWindow = selectedNetwork[
                (selectedNetwork['date'] >= window_start_date) & (selectedNetwork['date'] < window_end_date)]

            # select labeling data
            label_end_date = window_start_date + dt.timedelta(days=windowSize) + dt.timedelta(days=gap) + dt.timedelta(days=lableWindowSize)
            label_start_date = window_start_date + dt.timedelta(days=windowSize) + dt.timedelta(days=gap)
            selectedNetworkInLbelingWindow = selectedNetwork[
                (selectedNetwork['date'] >= label_start_date) & (selectedNetwork['date'] < label_end_date)]

           # generating the label for this window
           # 1 -> Increading Transactions 0 -> Decreasing Transactions
            label = 1 if (len(selectedNetworkInLbelingWindow) - len(selectedNetworkInGraphDataWindow)) > 0 else 0

            # group by for extracting node features
            outgoing_weight_sum = (selectedNetwork.groupby(by=['from'])['value'].sum())
            incoming_weight_sum = (selectedNetwork.groupby(by=['to'])['value'].sum())
            outgoing_count = (selectedNetwork.groupby(by=['from'])['value'].count())
            incoming_count = (selectedNetwork.groupby(by=['to'])['value'].count())

            # Populate graph with edges
            for item in selectedNetworkInGraphDataWindow.to_dict(orient="records"):
                from_node_features = {}
                to_node_features = {}
                # calculating node features for each edge
                # feature 1 -> sum of outgoing edge weights
                from_node_features["outgoing_edge_weight_sum"] = outgoing_weight_sum[item['from']]

                try:
                    to_node_features["outgoing_edge_weight_sum"] = outgoing_weight_sum[item['to']]
                except Exception as e:
                    to_node_features["outgoing_edge_weight_sum"] = 0

                # feature 2 -> sum of incoming edge weights
                to_node_features["incoming_edge_weight_sum"] = incoming_weight_sum[item['to']]
                try:
                    from_node_features["incoming_edge_weight_sum"] = incoming_weight_sum[item['from']]
                except Exception as e:
                    from_node_features["incoming_edge_weight_sum"] = 0
                # feature 3 -> number of outgoing edges
                from_node_features["outgoing_edge_count"] = outgoing_count[item['from']]
                try:
                    to_node_features["outgoing_edge_count"] = outgoing_count[item['to']]
                except Exception as e:
                    to_node_features["outgoing_edge_count"] = 0

                # feature 4 -> number of incoming edges
                to_node_features["incoming_edge_count"] = incoming_count[item['to']]
                try:
                    from_node_features["incoming_edge_count"] = incoming_count[item['from']]
                except Exception as e:
                    from_node_features["incoming_edge_count"] = 0

                # # feature 5 -> max outgoing edge weight
                # from_node_features["outgoing_edge_weight_max"] = selectedNetwork.groupby(by=['from'])['value'].max
                # # feature 6 -> max incoming edge weight
                # from_node_features["incoming_edge_weight_max"] = selectedNetwork.groupby(by=['from'])['value'].max
                # to_node_features["incoming_edge_weight_max"] = selectedNetwork.groupby(by=['from'])['value'].max
                # # feature 7 -> min outgoing edge weight
                # from_node_features["outgoing_edge_weight_min"] = selectedNetwork.groupby(by=['from'])['value'].min
                # to_node_features["outgoing_edge_weight_min"] = selectedNetwork.groupby(by=['from'])['value'].min
                # # feature 8 -> min incoming edge weight
                # from_node_features["incoming_edge_weight_min"] = selectedNetwork.groupby(by=['from'])['value'].min
                # to_node_features["incoming_edge_weight_min"] = selectedNetwork.groupby(by=['from'])['value'].min

                transactionGraph.add_nodes_from([(item["from"], from_node_features)])
                transactionGraph.add_nodes_from([(item["to"], to_node_features)])
                transactionGraph.add_edge(item["from"], item["to"], value=item["value"])
            featureNames = ["outgoing_edge_weight_sum", "incoming_edge_weight_sum", "outgoing_edge_count",
                            "incoming_edge_count"]
            window_start_date = window_start_date + dt.timedelta(days=1)
            directory = 'PygGraphs/TimeSeries/' + file
            if not os.path.exists(directory):
                os.makedirs(directory)
            pygData = self.from_networkx(transactionGraph, label=label, group_node_attrs=featureNames)
            with open(directory+"/" + file + "_" + "graph_"+str(indx), 'wb') as f:
                pickle.dump(pygData, f)

    def processDataDUration(self, file):
        # load each network file
        # Timer(2, functools.partial(self.exitfunc, file_path, file)).start()
        print("Processing {}".format(file))
        selectedNetwork = pd.read_csv((self.file_path + file), sep=' ', names=["from", "to", "date", "value"])

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

    def createNodeTokenNetworkCount(self, file, bucket):
        print("Processing {}".format(file))
        nodeHashMap = {}

        # try:
        #     with open(bucket + 'NodeTokenNetworkHashMap.txt') as f:
        #         nodeHashMap = ast.literal_eval(f.read())
        #         f.close()
        # except Exception as e:
        #     print(e)

        eachNetworkNodeHashMap = []
        selectedNetwork = pd.read_csv((self.file_path + bucket + "/" + file), sep=' ',
                                      names=["from", "to", "date", "value"])
        selectedNetwork = selectedNetwork.drop('value', axis=1)
        selectedNetwork = selectedNetwork.drop('date', axis=1)
        unique_node_ids = np.unique(selectedNetwork[['from', 'to']].values)
        id = 0
        for nodeID in unique_node_ids:
            # calculate the number of token participation
            id += 1
            # print("{} / {} inside file ".format(id , len(unique_node_ids)))
            if nodeID not in eachNetworkNodeHashMap:
                # first time we see this node ID in the current network
                if nodeID not in nodeHashMap:
                    nodeHashMap[nodeID] = 1
                else:
                    nodeHashMap[nodeID] += 1
                eachNetworkNodeHashMap.append(nodeID)
        with open(self.file_path + bucket + "/NodeCount/" + file.split(".")[0] + '_NodeTokenNetworkHashMap.txt',
                  'w+') as data:
            data.write(str(nodeHashMap))
            data.close()

    def multiprocessNodeNetworkCount(self):
        print("Process Started\n")
        self.processingIndx = 0

        master_list = ['bucket_1', 'bucket_2', 'bucket_3', 'bucket_4', 'bucket_5']
        with mp.Pool() as pool:
            results = pool.map(self.processBucketNodeNetworkCount, master_list, chunksize=1)

    def processBucketNodeNetworkCount(self, bucket):
        print("in bucket" + bucket + "\n")
        files = os.listdir(self.file_path + bucket)
        for file in files:
            if file.endswith(".txt"):
                self.createNodeTokenNetworkCount(file, bucket)
                shutil.move(self.file_path + bucket + "/" + file, self.file_path + bucket + "/Pr/" + file)

    def graphCreationHandler(self):
        files = os.listdir(self.file_path)
        for file in files:
            if file.endswith(".txt"):
                self.processingIndx += 1
                print("Processing {} / {} \n".format(self.processingIndx, len(files) - 3))
                p = Process(target=self.creatTimeSeriesGraphs, args=(file,))  # make process
                p.start()  # start function
                p.join(timeout=240)

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
                    shutil.move(self.file_path + file, self.file_path + "Processed/" + file)
                    print("Process finished successfully")
                    p.terminate()

        # stat_data.to_csv("final_data.csv")

    def graphCreationHandlerTimeSeries(self):
        files = os.listdir(self.timeseries_file_path)
        for file in files:
            if file.endswith(".txt"):
                print("Processing {} / {} \n".format(self.processingIndx, len(files) -2))
                p = Process(target=self.creatTimeSeriesGraphs, args=(file,))  # make process
                p.start()  # start function
                p.join(timeout=240)

                # Check if the process is still running
                if p.is_alive():
                    # The process is still running, terminate it
                    p.terminate()
                    print("The file is taking infinite time - check the file ")
                    shutil.move(self.timeseries_file_path + file, self.timeseries_file_path + "issue/" + file)
                    self.processingIndx += 1
                    print("Function timed out and was terminated")
                else:
                    # The process has finished
                    self.processingIndx += 1
                    shutil.move(self.timeseries_file_path + file, self.timeseries_file_path + "Processed/" + file)
                    print("Process finished successfully")
                    p.terminate()


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

    def read_and_merge_node_network_count(self):
        # Create an empty dictionary to store the merged result
        merged_dict = {}

        # Loop through all files with a .txt extension in the current directory
        indx = 0
        files = os.listdir(self.file_path + "NodeExistenceMatrix")
        for file in files:
            if file.endswith(".txt"):
                indx += 1
                print("Processing {}/{}".format(indx, len(files)))
                with open(self.file_path + "NodeExistenceMatrix/" + file, 'r') as f:
                    file_dict = eval(f.read())
                # Merge the dictionary with the merged result
                for key in file_dict:
                    if key in merged_dict:
                        merged_dict[key] += file_dict[key]
                    else:
                        merged_dict[key] = file_dict[key]

        # Print the merged dictionary
        with open(self.file_path + "FInal_NodeTokenNetworkHashMap.txt", 'w+') as data:
            data.write(str(merged_dict))
            data.close()

    def creat_hist_for_node_network_count(self):
        with open(self.file_path + "FInal_NodeTokenNetworkHashMap.txt", 'r') as f:
            file_dict = eval(f.read())
            # Create a list of the dictionary values
            values = list(file_dict.values())
            items_without_ones = {k: v for k, v in file_dict.items() if v != 1}
            values_without_one = list(items_without_ones.values())
            values_A_to_plot = [21 if i > 20 else i for i in values_without_one]
            # Define custom bin ranges
            bins = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, max(values_A_to_plot)]

            # Create the histogram
            plt.hist(values_A_to_plot, bins=bins, align='left')
            plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
            plt.xticks(rotation=45, ha='right')

            # Get the x-axis tick labels
            labels = [int(x) if x < 20 else "20+" for x in bins]

            # Set the tick labels
            plt.xticks(bins[:-1], labels[:-1])
            # Add labels and title
            plt.xlabel('Token network count')
            plt.ylabel('Number of nodes with that token network count')
            plt.title('Histogram of node counts')
            plt.savefig('histogram_token_count.png', dpi=300, bbox_inches="tight")
            # Show the plot
            plt.show()


if __name__ == '__main__':
    np = NetworkParser()
    np.graphCreationHandlerTimeSeries()
