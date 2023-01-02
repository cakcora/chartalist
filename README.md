<p align='center'>
  <img width='40%' src='https://chartalist.org/img/chartalist-hor.png' />
</p>

--------------------------------------------------------------------------------


# Chartalist
<p align='center'>
  <img width='100%' src='https://chartalist.org/images/Picture5.png' />
</p>

Please visit https://www.chartalist.org for more information.

## Overview
Chartalist is the first blockchain machine learning ready dataset platform from unspent transaction output and account-based blockchains.

The Chartalist package contains:
1. Dataloaders which automate and handle the download of datasets from a single package import and a simple two-argument function call.
2. Ability to use the downloaded dataset directly after download as a Pandas DataFrame from the same two-argument function call.
3. Graph makers for convenient generation of a NetworkX digraph from the network datasets.

## Installation
1. Download this repository and extract the contents to a desired location.
2. Inside the `chartalist_loader-main` folder will serve as the working directory.

### Requirements
Chartalist depends on the following:

- networkx>=2.8.3
- numpy>=1.22.3
- outdated>=0.2.1
- pandas>=1.4.2
- patool>=1.12
- requests>=2.27.1
- setuptools>=60.2.0
- torch>=1.11.0
- torch_scatter>=2.0.9


## Datasets

The following is a summary of the available datasets and their related tasks.  Use the corresponding version argument when using Chartalist to retrieve the correct dataset of interest.  Click on the dataset for more information. 

### Bitcoin ML-Ready Datasets
| Dataset                                      | Features                                                         | Version Constant                                |
| -------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------|
| [Ransomware Family: Bitcoinheist](https://chartalist.org/btc/TaskTypePrediction.html)              | address, year, day, length, weight, count, looped, neighbors, income, label                      | TYPE_PREDICTION                                 |       
| [Bitcoin Transaction Network Input](https://chartalist.org/BitcoinData.html)            | trans | TRANSACTION_NETWORK_INPUT_SAMPLE                   |
| [Bitcoin Transaction Network Output](https://chartalist.org/BitcoinData.html)           | trans | TRANSACTION_NETWORK_OUTPUT_SAMPLE                                   |
| [Bitcoin Block Times](https://chartalist.org/BitcoinData.html)      | unix_time | BLOCK_TIME                                   |
| [Bitcoin Price Data](https://chartalist.org/btc/TaskPriceAnalytics.html)                           | date, price, year, day, totaltx                                      | PRICE_PREDICTION                  |

### Ethereum ML-Ready Datasets
| Dataset                                      | Features                                                         | Version Constant                                |
| -------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------|
| [Ethereum Token Networks](https://chartalist.org/eth/TaskTypePrediction.html)                      | token_address, from_address, to_address, value, transaction_hash, log_index, block_number      |     TYPE_PREDICTION_TRANSACTIONS                                            |     
| [Ethereum Token Network Labels](https://chartalist.org/eth/TaskTypePrediction.html)                      | type, address, name     |     TYPE_PREDICTION_LABELS  |  
| [EtherDelta Ether-to-Token Transactions](https://chartalist.org/eth/TaskPatternDetection.html)                  | transaction_hash, block_number, timestamp,	tokenGet,	amountGet,	tokenGive,	amountGive,	get,	give                      |            ANOMALY_DETECTION_ETHER_DELTA_TRADES            |
| [IDEX Ether-to-Token Transactions](https://chartalist.org/eth/TaskPatternDetection.html)       | transaction_hash, status, block_number, gas, gas_price, timestamp, amountBuy, amountSell, expires, nonce, amount, tradeNonce, feeMake, feeTake, tokenBuy, tokenSell, maker, taker  |        ANOMALY_DETECTION_IDEX       |
| [Ether-to-Token Ether-Dollar Price](https://chartalist.org/eth/TaskPatternDetection.html)            | Date(UTC), UnixTimeStamp, Value   |            ANOMALY_DETECTION_ETHER_DOLLAR_PRICE                                     |
| [Bytom Network](https://chartalist.org/eth/TaskMultilayer.html)                                | fromAddress, toAddress, time, amount                                          | MULTILAYER_BYTOM                                |
| [Cybermiles Network](https://chartalist.org/eth/TaskMultilayer.html)                           | fromAddress, toAddress, time, amount                                          | MULTILAYER_CYBERMILES                           |
| [Decentraland Network](https://chartalist.org/eth/TaskMultilayer.html)                         | fromAddress, toAddress, time, amount                                          | MULTILAYER_DECENTRALAND                         |
| [Tierion Network](https://chartalist.org/eth/TaskMultilayer.html)                              | fromAddress, toAddress, time, amount                                          | MULTILAYER_TIERION                              |
| [Vechain Network](https://chartalist.org/eth/TaskMultilayer.html)                              | fromAddress, toAddress, time, amount                                          | MULTILAYER_VECHAIN                              |
| [ZRX Network](https://chartalist.org/eth/TaskMultilayer.html)                                  | fromAddress, toAddress, time, amount                                          | MULTILAYER_ZRX                                  |
| [Ethereum VeChain Token Transactions](https://chartalist.org/eth/TaskPriceAnalytics.html)                   | fromAddress, toAddress, time, amount                            |                   PRICE_PREDICTION_VECHAIN                              |
| [Ethereum ZRX Token Transactions](https://chartalist.org/eth/TaskPriceAnalytics.html)                   | fromAddress, toAddress, time, amount                              |              PRICE_PREDICTION_ZRX                                   |
| [Stablecoin ERC20 Transactions](https://chartalist.org/eth/StablecoinAnalysis.html)                | fromAddress, toAddress, time, amount                                         |   STABLECOIN_ERC20                                              |

### Dashcoin ML-Ready Datasets
| Dataset                                      | Features                                                         | Version Constant                                |
| -------------------------------------------- | -------------------------------------------------------------- | ------------------------------------------------|     
| [Dashcoin Transaction Network Input](https://chartalist.org/dash/DashData.html)      | trans                     | TRANSACTION_NETWORK_INPUT_SAMPLE                                                          |
| [Dashcoin Transaction Network Output](https://chartalist.org/dash/DashData.html)      | trans                     | TRANSACTION_NETWORK_OUTPUT_SAMPLE                                                        |

## Using Chartalist
1. Navigate to the folder `chartalist_loader-main` and create a new `.py` script or add one which will serve as the working environment.
2. Ensure to add `import chartalist` at the top of the script.
```py 
import chartalist 
```
3. All datasets in Chartalist can be downloaded and referenced as a Pandas DataFrame in a single function call.

For example:

```py
data = chartalist.get_dataset(dataset='dashcoin', version='chartalist.DashcoinLoader.TRANSACTION_NETWORK_OUTPUT_SAMPLE', download=True, data_frame=True)
```
There are currently three options for the dataset argument:
- ethereum
- bitcoin  
- dashcoin

Depending on the choice of the dataset argument, the version argument will take the following format:

For ethereum:
```py
version=chartalist.EthereumLoader.
```
For bitcoin:
```py
version=chartalist.BitcoinLoaders.
```
For dashcoin:
```py
version=chartalist.DashcoinLoader.
```

Refer to [#Datasets](#datasets) for the appropriate constant to append to the end of the version above and then the function is now ready to be used.

4. Upon execution of the function, the corresponding dataset will be downloaded under the `data` folder in the working directory, if not already downloaded, when the script is executed and the Pandas DataFrame containing the dataset can be used directly for processing.

> **_NOTE:_**  Due the large nature of certain datasets, only sample data will be downloaded by the dataloader. If the complete dataset is required, click on the link corresponding to the dataset of interest and manually download the data from our website.  Replace the contents of the sample dataset with the contents of the complete dataset under the `data` folder and proceed as normal.

## Generating Networks

The Bitcoin and Dashcoin Transaction Network Input and Output datasets require the use of a Chartalist graph maker to be converted into a usable NetworkX digraph.  See `bitcoin_network_example.py` or `dashcoin_network_example.py` for instructions.

For other Network datasets that have labels fromAddress, toAddress, and value labels such as the Ethereum Token Network dataset, the generation of a Networkx digraph can be done directly.  See `ethereum_network_example.py` for instructions.

## Parsing Datasets

Parsing any dataset for basic statistical information can be done so easily by using the Pandas Dataframe returned by the dataloader. See `stablecoin_erc20_example.py` for reference.

##  Address Exclusion
Please use <a href="https://www.chartalist.org/AddressExclusion.html"> our online tool </a> to submit your request for removing an address from our dataset due to security and privacy issue.

## BibTeX Citation

If you use Chartalist in a scientific publication, please cite us with the following bibtex:

```
@article{Chartalist2022,
    year      = {2022},
    author    = {Kiarash Shamsi and Friedhelm Victor and Murat Kantarcioglu and Yulia R. Gel and Cuneyt G. Akcora},
    title     = {Chartalist: Labeled Graph Datasets for UTXO and Account-based Blockchains},
    journal   = {36th Conference on Neural Information Processing Systems (NeurIPS 2022) }
    volume    = {36},
    pages     = {1--10}
}