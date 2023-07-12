import shutil
import networkx as nx
import pandas as pd
import datetime as dt

def creatNetworkGraphs(self, file):
    print("Processing {}".format(file))
    selectedNetwork = pd.read_csv((self.file_path + file), sep=' ', names=["from", "to", "date", "value"])
    selectedNetwork['date'] = pd.to_datetime(selectedNetwork['date'], unit='s').dt.date
    selectedNetwork['value'] = selectedNetwork['value'].astype(float)


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

            transactionGraph.add_nodes_from([(item["from"], from_node_features)])
            transactionGraph.add_nodes_from([(item["to"], to_node_features)])
            transactionGraph.add_edge(item["from"], item["to"], value=item["value"])

        featureNames = ["outgoing_edge_weight_sum", "incoming_edge_weight_sum", "outgoing_edge_count",
                        "incoming_edge_count"]

        return transactionGraph

