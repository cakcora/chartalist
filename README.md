# Chartalist
Please visit https://www.chartalist.org for more information.

## Overview
Chartalist is the first blockchain machine learning ready dataset platform from unspent transaction output and account-based blockchains.

The Chartalist package contains:
1. Dataloaders which automate and handle the download of datasets from a single package import and a simple two-argument function call.
2. Ability to use the downloaded dataset directly after download as a Pandas DataFrame from the same two-argument function call.

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
| Dataset                                      | Labels                                                         | Version Argument                                |
| -------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------|
| [Ransomware Family: Bitcoinheist](https://chartalist.org/btc/TaskTypePrediction.html)              | address, year, day, length, weight, count, looped, neighbors, income, label                      | type_prediction                                 |       
| [Bitcoin Transaction Network Input](https://chartalist.org/BitcoinData.html)            | trans | trans_net_in                   |
| [Bitcoin Transaction Network Output](https://chartalist.org/BitcoinData.html)           | trans | trans_net_out                                   |
| [Bitcoin Block Times](https://chartalist.org/BitcoinData.html)      | unix_time | block_time                                   |
| [Bitcoin Price Data](https://chartalist.org/btc/TaskPriceAnalytics.html)                           | date, price, year, day, totaltx                                      | price_prediction                  |

### Ethereum ML-Ready Datasets
| Dataset                                      | Labels                                                         | Version Argument                                |
| -------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------|
| [Ethereum Token Networks](https://chartalist.org/eth/TaskTypePrediction.html)                      | token_address, from_address, to_address, value, transaction_hash, log_index, block_number      |     type_prediction_trans                                            |     
| [Ethereum Token Network Labels](https://chartalist.org/eth/TaskTypePrediction.html)                      | type, address, name     |     type_prediction_labels  |  
| [EtherDelta Ether-to-Token Transactions](https://chartalist.org/eth/TaskPatternDetection.html)                  | transaction_hash, block_number, timestamp,	tokenGet,	amountGet,	tokenGive,	amountGive,	get,	give                      |            anomaly_detection_ether_delta_trades            |
| [IDEX Ether-to-Token Transactions](https://chartalist.org/eth/TaskPatternDetection.html)       | transaction_hash, status, block_number, gas, gas_price, timestamp, amountBuy, amountSell, expires, nonce, amount, tradeNonce, feeMake, feeTake, tokenBuy, tokenSell, maker, taker  |        anomaly_detection_idex       |
| [Ether-to-Token Ether-Dollar Price](https://chartalist.org/eth/TaskPatternDetection.html)            | Date(UTC), UnixTimeStamp, Value   |            anomaly_detection_ether_dollar_price                                     |
| [Bytom Network](https://chartalist.org/eth/TaskMultilayer.html)                                | fromAddress, toAddress, time, amount                                          | multilayer_bytom                                |
| [Cybermiles Network](https://chartalist.org/eth/TaskMultilayer.html)                           | fromAddress, toAddress, time, amount                                          | multilayer_cybermiles                           |
| [Decentraland Network](https://chartalist.org/eth/TaskMultilayer.html)                         | fromAddress, toAddress, time, amount                                          | multilayer_decentraland                         |
| [Tierion Network](https://chartalist.org/eth/TaskMultilayer.html)                              | fromAddress, toAddress, time, amount                                          | multilayer_tierion                              |
| [Vechain Network](https://chartalist.org/eth/TaskMultilayer.html)                              | fromAddress, toAddress, time, amount                                          | multilayer_vechain                              |
| [ZRX Network](https://chartalist.org/eth/TaskMultilayer.html)                                  | fromAddress, toAddress, time, amount                                          | multilayer_zrx                                  |
| [Ethereum VeChain Token Transactions](https://chartalist.org/eth/TaskPriceAnalytics.html)                   | fromAddress, toAddress, time, amount                            |                   price_prediction_vechain                              |
| [Ethereum ZRX Token Transactions](https://chartalist.org/eth/TaskPriceAnalytics.html)                   | fromAddress, toAddress, time, amount                              |              price_prediction_zrx                                   |
| [Stablecoin ERC20 Transactions](https://chartalist.org/eth/StablecoinAnalysis.html)                | fromAddress, toAddress, time, amount                                         |   stablecoin_erc20                                              |

### Dashcoin ML-Ready Datasets
| Dataset                                      | Labels                                                         | Version Argument                                |
| -------------------------------------------- | -------------------------------------------------------------- | ------------------------------------------------|     
| [Dashcoin Transaction Network Input](https://chartalist.org/dash/DashData.html)      | trans                     | trans_net_in                                                          |
| [Dashcoin Transaction Network Output](https://chartalist.org/dash/DashData.html)      | trans                     | trans_net_out                                                        |

## Using Chartalist
1. Navigate to the folder `chartalist_loader-main` and create a new `.py` script or add one which will serve as the working environment.
2. Ensure to add `import chartalist` at the top of the script.
```py 
import chartalist 
```
3. All datasets in Chartalist can be downloaded and referenced as a Pandas DataFrame in a single function call.

For example:

```py
data = chartalist.get_dataset(dataset='ethereum', version='trans_net', download=True, data_frame=True)
```
There are currently three options for the dataset argument:
- ethereum
- bitcoin  
- dashcoin

Depending on the choice of the dataset argument, please refer to [#Datasets](#datasets) for the appropriate version argument.

4. The corresponding dataset will be downloaded under the `data` folder in the working directory if not already when the script is ran and the Pandas DataFrame containing the dataset can be used directly for processing.

> **_NOTE:_**  Due the large nature of certain datasets, only sample data will be downloaded by the dataloader. If the complete dataset is required, click on the link corresponding to the dataset of interest and manually download the data from our website.  Replace the complete dataset with the sample dataset file under `data` and proceed as normal.

## Using Example Scripts

The included example scripts are provided as a starting point for processing the data in the datasets and meant to help familiarize the new user with the data format.
