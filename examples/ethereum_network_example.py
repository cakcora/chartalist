import chartalist
import networkx as nx

'''
    The following script shows an example of how to generate a NetworkX Digraph 
    from the Ethereum Token Networks dataset.
'''

def main():
    print("Started\n")

    # Retrieve dataset by call to dataloader
    # Ethereum Network Dataset
    etherNetwork = chartalist.get_dataset(dataset='ethereum',
                                                     version=chartalist.EthereumLoader.TYPE_PREDICTION_TRANSACTIONS, download=True,
                                                     data_frame=True)

    G = nx.DiGraph()

    # Populate graph with edges
    for item in etherNetwork.to_dict(orient="records"):
        G.add_edge(item["from_address"], item["to_address"], value=item["value"])

    # Digraph G can now be used directly for processing

    print("\nFinished \n")


if __name__ == '__main__':
    main()
