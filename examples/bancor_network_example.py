import math

import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
import pygraphviz
import chartalist
import networkx as nx

'''
    The following script shows an example of how to generate a NetworkX Digraph 
    and plot that from the Bancor token networks dataset.
'''


def main():
    print("Started\n")

    # Retrieve dataset by call to dataloader
    # Ethereum Network Dataset
    etherNetwork = chartalist.get_dataset(dataset='ethereum',
                                          version=chartalist.EthereumLoader.TRANSACTION_NETWORK_BANCOR, download=True,
                                          data_frame=True)

    etherNetwork = etherNetwork.sort_values(by=['time'], ascending=True)

    startTime = etherNetwork.iloc[0]["time"]

    # creating graph for 6, 12, 18 and 24 hours periods, respectively
    for duration in [6, 12, 18, 24]:

        endTime = startTime + (60 * 60 * duration)

        timeData = etherNetwork[(etherNetwork["time"] > startTime) & (etherNetwork["time"] < endTime)]

        G = nx.DiGraph()

        # Populate graph with edges
        for item in timeData.to_dict(orient="records"):
            item = {x.replace(' ', ''): v for x, v in item.items()}
            print(item)
            G.add_edge(item["fromAddress"], item["toAddress"], value=item["amount"])
        # Digraph G can now be used directly for processing


        # Visualize graph with different modes
        plt.figure(figsize=(12, 16))
        pos = graphviz_layout(G)
        nx.draw_networkx_nodes(G, pos, nodelist=G.nodes(), node_size=140, linewidths=0.2, vmin=0, vmax=1)
        nx.draw_networkx_edges(G, pos, edgelist=G.edges(), width=0.8, edge_color="black")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig("figs/Bancor_manual__" + str(duration) + ".svg" ,  format='svg', dpi=1200)
        plt.show()

        for mode in ["dot", "neato", "fdp", "sfdp", "twopi", "circo"]:
            pos = nx.nx_agraph.graphviz_layout(G, prog=mode)
            # nx.spring_layout(G, k=0.2 , iterations=20)
            nx.draw(G, arrows=True, node_size=50, pos=pos)
            plt.savefig("figs/" + mode + "__" + str(duration) + ".svg" ,  format='svg', dpi=1200)

            plt.show()

    print("\nFinished \n")


if __name__ == '__main__':
    main()
