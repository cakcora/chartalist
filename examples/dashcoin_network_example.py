import chartalist
from chartalist.common.dashcoin_graph_maker import DashcoinGraphMaker

'''
    The following script shows an example of how to generate a NetworkX digraph from the 
    Dashcoin Transaction Network Input and Output dataset using the Chartalist DashcoinGraphMaker.
'''

def main():
    print("Started\n")

    # Retrieve datasets by call to dataloader
    # Dashcoin Transaction Network Input and Output Dataframes
    dash_in = chartalist.get_dataset(dataset='dashcoin',
                                                     version=chartalist.DashcoinLoader.TRANSACTION_NETWORK_INPUT_SAMPLE, download=True,
                                                     data_frame=True)
    dash_out = chartalist.get_dataset(dataset='dashcoin',
                                                     version=chartalist.DashcoinLoader.TRANSACTION_NETWORK_OUTPUT_SAMPLE, download=True,
                                                     data_frame=True)

    # Create an instance of the graph maker
    dgm = DashcoinGraphMaker()

    # Call the make_graph function to return a NetworkX digraph
    G = dgm.make_graph(dash_in, dash_out)

    # Digraph G can now be used directly for processing

    print("\nFinished \n")


if __name__ == '__main__':
    main()
