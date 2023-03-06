import functools
import os
import shutil
import sys
from multiprocessing import Process
from threading import Timer

import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt
import datetime as dt

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

            num_nodes = len(transactionGraphs.nodes())
            num_edges = len(transactionGraphs.edges())
            density = nx.density(transactionGraphs)
            transactionGraphs = transactionGraphs.to_undirected()
            if (nx.is_connected(transactionGraphs)):
                diameter = nx.diameter(transactionGraphs)
                avg_shortest_path_length = nx.average_shortest_path_length(transactionGraphs)
                clique_number = nx.graph_clique_number(transactionGraphs)
            else:
                Gcc = sorted(nx.connected_components(transactionGraphs), key=len, reverse=True)
                biggestConnectedComponent = transactionGraphs.subgraph(Gcc[0])
                diameter = nx.diameter(biggestConnectedComponent)
                avg_shortest_path_length = nx.average_shortest_path_length(biggestConnectedComponent)
                clique_number = nx.graph_clique_number(biggestConnectedComponent)

            # try:
            #     diameter = max([max(j.values()) for (i,j) in nx.shortest_path_length(transactionGraphs)])
            # except Exception as e:
            #     diameter = "Error"
            #
            # try:
            #     avg_shortest_path_length = nx.average_shortest_path_length(transactionGraphs)
            # except Exception as e:
            #     avg_shortest_path_length = "NC"

            max_degree_centrality = max(nx.degree_centrality(transactionGraphs).values())
            min_degree_centrality = min(nx.degree_centrality(transactionGraphs).values())
            max_closeness_centrality = max(nx.closeness_centrality(transactionGraphs).values())
            min_closeness_centrality = min(nx.closeness_centrality(transactionGraphs).values())
            max_betweenness_centrality = max(nx.betweenness_centrality(transactionGraphs).values())
            min_betweenness_centrality = min(nx.betweenness_centrality(transactionGraphs).values())
            # max_eigenvector_centrality = max(nx.eigenvector_centrality(transactionGraphs).values())
            # min_eigenvector_centrality = min(nx.eigenvector_centrality(transactionGraphs).values())
            # pagerank = nx.pagerank(transactionGraphs)
            assortativity = nx.degree_assortativity_coefficient(transactionGraphs)
            # try:
            #     clique_number = nx.graph_clique_number(transactionGraphs)
            # except Exception as e:
            #     clique_number = "NC"
            motifs = nx.algorithms.community.kernighan_lin_bisection(transactionGraphs, weight='weight')

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
            # nx.draw(transactionGraphs, node_size=10)
            # plt.savefig("images/" + file + "_" + str(timeFrame) + "_" + ".png")

            print("\nFinisheng processing {} \n".format(file + "   " + str(timeFrame)))

    def main(self):
        print("Process Started\n")
        self.processingIndx = 0
        files = os.listdir(self.file_path)
        for file in files:
            if file.endswith(".txt"):
                print("Processing {} / {} \n".format(self.processingIndx, len(files)-3))
                p = Process(target=self.processGraph, args=(file,))  # make process
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



if __name__ == '__main__':
    np = NetworkParser()
    np.main()
