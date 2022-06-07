# Chartalist
Please visit https://www.chartalist.org for more information.

## Overview
Chartalist is the first blockchain machine learning ready dataset platform from unspent transaction output and account-based blockchains.

The Chartalist package contains:

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

The following is a summary of the available datasets and their related tasks.  Use the corresponding version argument when using Chartalist to retrieve the correct dataset of interest.  Network datasets with multiple version arguments 

### Bitcoin ML-Ready Datasets
| Dataset                                      | Task                                                         | Version Argument                                |
| -------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------|
| Ransomware Family: Bitcoinheist              | Address and transaction type prediction                      | type_prediction                                 |       
| Bitcoin Transaction Network Input            | Anomalous transaction pattern detection & Address clustering | trans_net_in & trans_net_out                    |
| Bitcoin Transaction Network Output           | Anomalous transaction pattern detection & Address clustering | trans_net_out                                   |
| Bitcoin Price Data                           | Bitcoin Price Analytics                                      | price_prediction                                |

### Ethereum ML-Ready Datasets
| Dataset                                      | Task                                                         | Version Argument                                |
| -------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------|
| Ethereum Token Networks                      | Address and transaction type prediction                      |                                                 |       
| Ether-to-Token Transactions                  | Anomalous transaction pattern detection                      |                                                 |
| Bytom Network                                | Multilayer analysis                                          | multilayer_bytom                                |
| Cybermiles Network                           | Multilayer analysis                                          | multilayer_cybermiles                           |
| Decentraland Network                         | Multilayer analysis                                          | multilayer_decentraland                         |
| Tierion Network                              | Multilayer analysis                                          | multilayer_tierion                              |
| Vechain Network                              | Multilayer analysis                                          | multilayer_vechain                              |
| ZRX Network                                  | Multilayer analysis                                          | multilayer_zrx                                  |
| Ethereum Token Tansactions                   | Ethereum token price prediction                              |                                                 |
| Stablecoin ERC20 Transactions                | Stable coin analysis                                         |                                                 |

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
There are two options for the dataset argument:
- ethereum
- bitcoin  

Depending on the choice of the dataset argument, the following are options for the version argument:

Ethereum:
- trans_net
- multilayer_bytom
- multilayer_cybermiles
- multilayer_decentraland
- multilayer_tierion
- multilayer_vechain
- multilayer_zrx  

Bitcoin:
- trans_net_in
- trans_net_out
- price_prediction
- type_prediction

4. The corresponding dataset will be downloaded if not already when the script is ran and the Panda DataFrame can be used directly for processing.

## Using Example Scripts

The included example scripts are provided as a starting point for processing the data in the datasets and meant to help familiarize the new user with the data format.
