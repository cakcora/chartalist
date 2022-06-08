import chartalist
from chartalist.common.bitcoin_graph_maker import BitcoinGraphMaker

'''
    The following script shows an example of how to generate a NetworkX digraph from the 
    Bitcoin Transaction Network Input and Output dataset using the Chartalist BitcoinGraphMaker.
'''

def main():
    print("Started\n")

    # Retrieve datasets by call to dataloader
    # Bitcoin Transaction Network Input and Output Dataframes
    btc_in = chartalist.get_dataset(dataset='bitcoin',
                                                     version=chartalist.BitcoinLoaders.TRANSACTION_NETWORK_INPUT_SAMPLE, download=True,
                                                     data_frame=True)
    btc_out = chartalist.get_dataset(dataset='bitcoin',
                                                     version=chartalist.BitcoinLoaders.TRANSACTION_NETWORK_OUTPUT_SAMPLE, download=True,
                                                     data_frame=True)

    # Create an instance of the graph maker
    bgm = BitcoinGraphMaker()

    # Call the make_graph function to return a NetworkX digraph
    G = bgm.make_graph(btc_in, btc_out)

    # Digraph G can now be used directly for processing

    print("\nFinished \n")


if __name__ == '__main__':
    main()
