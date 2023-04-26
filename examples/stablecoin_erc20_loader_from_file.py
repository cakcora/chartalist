import networkx as nx
import pandas as pd

'''
    The following script shows an example of how to parse through the data in the 
    Chartalist Stablecoin ERC20 Transactions dataset to obtain some basic statistical information and creat transaction graph.
    
'''


def main():
    print("Started\n")

    # Retrieve dataset by call to dataloader
    # Ethereum Stable Coin ERC20
    ethereumStableCoinERC20 = pd.read_csv('token_transfers_V3.0.0.csv', sep=',')

    # Top 6 stablecoins plus WLUNA
    stablecoin = {
        "tether": ("0xdac17f958d2ee523a2206206994597c13d831ec7".lower()),
        "usdc": ("0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48".lower()),
        "dai": ("0x6b175474e89094c44da98b954eedeac495271d0f".lower()),
        "terrausd": ("0xa47c8bf37f92aBed4A126BDA807A7b7498661acD".lower()),
        "pax": ("0x8e870d67f660d95d5be530380d0ec0bd388289e1".lower()),
        "wluna": ("0xd2877702675e6cEb975b4A1dFf9fb7BAF4C91ea9".lower())
    }

    # Creating trasaction graph
    transactionGraphs = {
        "tether": nx.DiGraph(),
        "usdc": nx.DiGraph(),
        "dai": nx.DiGraph(),
        "terrausd": nx.DiGraph(),
        "pax": nx.DiGraph(),
        "wluna": nx.DiGraph()
    }

    # Populate graph with edges
    for item in ethereumStableCoinERC20.to_dict(orient="records"):
        if item["contract_address"] == stablecoin["tether"]:
            transactionGraphs["tether"].add_edge(item["from_address"], item["to_address"], value=item["value"])
        elif item["contract_address"] == stablecoin["usdc"]:
            transactionGraphs["usdc"].add_edge(item["from_address"], item["to_address"], value=item["value"])
        elif item["contract_address"] == stablecoin["dai"]:
            transactionGraphs["dai"].add_edge(item["from_address"], item["to_address"], value=item["value"])
        elif item["contract_address"] == stablecoin["terrausd"]:
            transactionGraphs["terrausd"].add_edge(item["from_address"], item["to_address"], value=item["value"])
        elif item["contract_address"] == stablecoin["pax"]:
            transactionGraphs["pax"].add_edge(item["from_address"], item["to_address"], value=item["value"])
        elif item["contract_address"] == stablecoin["wluna"]:
            transactionGraphs["wluna"].add_edge(item["from_address"], item["to_address"], value=item["value"])

    # Print Number of Nodes and Edges
    for graph in transactionGraphs:
        print("{} graph has {} edges and {} nodes \n".format(graph, transactionGraphs[graph].number_of_edges(),
                                                             transactionGraphs[graph].number_of_nodes()))

    # Track number of transactions per timeWindow
    dailyTransactions = []
    for i in range(0, 214):
        dailyTransactions.append({
            stablecoin["tether"]: 0,
            stablecoin["usdc"]: 0,
            stablecoin["dai"]: 0,
            stablecoin["terrausd"]: 0,
            stablecoin["pax"]: 0,
            stablecoin["wluna"]: 0,
        })

    # Counting daily transaction for each network
    # first timeWindow in data as UNIX timestamp = 86400 sec
    for item in ethereumStableCoinERC20.to_dict(orient="records"):
        try:
            dailyTransactions[int((int(item['time_stamp']) - 1648811666) / 86400)][item['contract_address']] += 1
        except Exception as e:
            print(e)
            print(int((int(item['time_stamp']) - 1648811666) / 86400))
            print(item['contract_address'])

    # From Thursday, April 1, 2022 to Tuesday, November 1, 2022
    print("\nNumber of transactions per timeWindow")
    dayVal = 0
    for day in dailyTransactions:
        print(str(dayVal) + ": " + str(dailyTransactions[dayVal]) + " transactions")
        dayVal += 1
        print("\n")

    print("\nFinished \n")

if __name__ == '__main__':
    main()
