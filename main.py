import matplotlib.pyplot as plt
import networkx as nx
import chartalist
from chartalist.common.bitcoin_graph_maker import BitcoinGraphMaker
from chartalist.common.dashcoin_graph_maker import DashcoinGraphMaker


def main():
    print("Started\n")
    #
    # Ethereum Transaction Network
    ethereumTn = chartalist.get_dataset(dataset='ethereum', version=chartalist.EthereumLoader.TRANSACTION_NETWORK_BANCOR,
                                        download=True, data_frame=True)

    # Ethereum Multilayer Task
    ethereumBytomTn = chartalist.get_dataset(dataset='ethereum', version=chartalist.EthereumLoader.MULTILAYER_BYTOM,
                                             download=True, data_frame=True)

    ethereumCybermilesTn = chartalist.get_dataset(dataset='ethereum',
                                                  version=chartalist.EthereumLoader.MULTILAYER_CYBERMILES,
                                                  download=True,
                                                  data_frame=True)

    ethereumDecentralandTn = chartalist.get_dataset(dataset='ethereum',
                                                    version=chartalist.EthereumLoader.MULTILAYER_DECENTRALAND,
                                                    download=True,
                                                    data_frame=True)

    ethereumTierionTn = chartalist.get_dataset(dataset='ethereum', version=chartalist.EthereumLoader.MULTILAYER_TIERION,
                                               download=True,
                                               data_frame=True)
    ethereumVechainTn = chartalist.get_dataset(dataset='ethereum', version=chartalist.EthereumLoader.MULTILAYER_VECHAIN,
                                               download=True,
                                               data_frame=True)

    ethereumZrxTn = chartalist.get_dataset(dataset='ethereum', version=chartalist.EthereumLoader.MULTILAYER_ZRX,
                                           download=True,
                                           data_frame=True)

    # Ethereum Stable Coin ERC20
    ethereumStableCoinERC20 = chartalist.get_dataset(dataset='ethereum',
                                                     version=chartalist.EthereumLoader.STABLECOIN_ERC20, download=True,
                                                     data_frame=True)

    # Ethereum Type Prediction
    ethereumTypePredictionTn = chartalist.get_dataset(dataset='ethereum',
                                                      version=chartalist.EthereumLoader.TYPE_PREDICTION_LABELS,
                                                      download=True, data_frame=True)

    ethereumTypePredictionLables = chartalist.get_dataset(dataset='ethereum',
                                                          version=chartalist.EthereumLoader.TYPE_PREDICTION_TRANSACTIONS,
                                                          download=True, data_frame=True)

    # Ethereum Price Prediction
    ethereumTypePricePredictionZRX = chartalist.get_dataset(dataset='ethereum',
                                                          version=chartalist.EthereumLoader.PRICE_PREDICTION_ZRX,
                                                          download=True, data_frame=True)

    ethereumTypePricePredictionVechain = chartalist.get_dataset(dataset='ethereum',
                                                          version=chartalist.EthereumLoader.PRICE_PREDICTION_VECHAIN,
                                                          download=True, data_frame=True)

    # Ethereum anomaly detection
    ethereumANomalyDetectionEDP = chartalist.get_dataset(dataset='ethereum',
                                                                version=chartalist.EthereumLoader.ANOMALY_DETECTION_ETHER_DOLLAR_PRICE,
                                                                download=True, data_frame=True)
    ethereumANomalyDetectionEDT = chartalist.get_dataset(dataset='ethereum',
                                                                version=chartalist.EthereumLoader.ANOMALY_DETECTION_ETHER_DELTA_TRADES,
                                                                download=True, data_frame=True)
    ethereumANomalyDetectionIDEX = chartalist.get_dataset(dataset='ethereum',
                                                                version=chartalist.EthereumLoader.ANOMALY_DETECTION_IDEX,
                                                                download=True, data_frame=True)


    # Bitcoin Trans Net
    bitcoinTnOut = chartalist.get_dataset(dataset='bitcoin',
                                          version=chartalist.BitcoinLoaders.TRANSACTION_NETWORK_OUTPUT_SAMPLE,
                                          download=True, data_frame=True)

    bitcoinTnIn = chartalist.get_dataset(dataset='bitcoin',
                                         version=chartalist.BitcoinLoaders.TRANSACTION_NETWORK_INPUT_SAMPLE,
                                         download=True, data_frame=True)

    # Bitcoin Graph Maker Example
    graphMaker = BitcoinGraphMaker()
    G = graphMaker.make_graph(in_df=bitcoinTnIn, out_df=bitcoinTnOut)
    colormap = graphMaker.get_graph_color_map(G)
    nx.draw(G, node_color=colormap, with_labels=True)
    plt.show()
    print(G)

    # Bitcoin Price Prediction
    bitcoinTnPricePredict = chartalist.get_dataset(dataset='bitcoin',
                                                   version=chartalist.BitcoinLoaders.PRICE_PREDICTION, download=True,
                                                   data_frame=True)

    # Bitcoin Type Prediction
    bitcoinTnTypePredict = chartalist.get_dataset(dataset='bitcoin', version=chartalist.BitcoinLoaders.TYPE_PREDICTION,
                                                  download=True,
                                                  data_frame=True)

    # Bitcoin Block Time
    bitcoinBlockTime = chartalist.get_dataset(dataset='bitcoin', version=chartalist.BitcoinLoaders.BLOCK_TIME,
                                              download=True,
                                              data_frame=True)
    # Dashcoin Trans Net
    dashcoinTnOut = chartalist.get_dataset(dataset='dashcoin',
                                           version=chartalist.DashcoinLoader.TRANSACTION_NETWORK_OUTPUT_SAMPLE,
                                           download=True,
                                           data_frame=True)
    dashcoinTnIn = chartalist.get_dataset(dataset='dashcoin',
                                          version=chartalist.DashcoinLoader.TRANSACTION_NETWORK_INPUT_SAMPLE,
                                          download=True,
                                          data_frame=True)

    # Bitcoin Graph Maker Example
    graphMaker = DashcoinGraphMaker()
    G = graphMaker.make_graph(in_df=dashcoinTnIn, out_df=dashcoinTnOut)
    colormap = graphMaker.get_graph_color_map(G)
    nx.draw(G, node_color=colormap, with_labels=True)
    plt.show()
    chartalist.get_info()


    print("Finished \n")


if __name__ == '__main__':
    main()
