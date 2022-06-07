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

-	numpy>=1.19.1
-	ogb>=1.2.6
-	outdated>=0.2.0
-	pandas>=1.1.0
-	pillow>=7.2.0
-	ogb>=1.2.6
-	pytz>=2020.4
-	torch>=1.7.0
-	torchvision>=0.8.2
-	tqdm>=4.53.0
-	scikit-learn>=0.20.0
-	scipy>=1.5.4


## Datasets

The following is a summary of the available datasets and their related tasks.  Use the corresponding version argument when using Chartalist to retrieve the correct dataset of interest.  Click on the dataset for more information. 

### Bitcoin ML-Ready Datasets
| Dataset                                      | Labels                                                         | Version Argument                                |
| -------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------|
| [Ransomware Family: Bitcoinheist](https://chartalist.org/btc/TaskTypePrediction.html)              | address, year, day, length, weight, count, looped, neighbors, income, label                      | type_prediction                                 |       
| [Bitcoin Transaction Network Input](https://chartalist.org/BitcoinData.html)            | trans | trans_net_in                   |
| [Bitcoin Transaction Network Output](https://chartalist.org/BitcoinData.html)           | trans | trans_net_out                                   |
| [Bitcoin Price Data](https://chartalist.org/btc/TaskPriceAnalytics.html)                           | date, price, year, day, totaltx                                      | price_prediction                                |

### Ethereum ML-Ready Datasets
| Dataset                                      | Labels                                                         | Version Argument                                |
| -------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------|
| [Ethereum Token Networks](https://chartalist.org/eth/TaskTypePrediction.html)                      | token_address, from_address, to_address, value, transaction_hash, log_index, block_number      |     type_prediction_trans                                            |     
| [Ethereum Token Network Labels](https://chartalist.org/eth/TaskTypePrediction.html)                      | type, address, name     |     type_prediction_trans  |  
| [Ether-to-Token Transactions](https://chartalist.org/eth/TaskPatternDetection.html)                  | Anomalous transaction pattern detection                      |                                                 |
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
There are three options for the dataset argument:
- ethereum
- bitcoin  
- dashcoin

Depending on the choice of the dataset argument, please refer to [#Datasets](datasets) for the appropriate version argument.

4. The corresponding dataset will be downloaded under the `data` folder in the working directory if not already when the script is ran and the Panda DataFrame can be used directly for processing.

## Using Example Scripts

The included example scripts are provided as a starting point for processing the data in the datasets and meant to help familiarize the new user with the data format.
