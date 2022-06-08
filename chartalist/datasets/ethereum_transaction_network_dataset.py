import csv
import os
from typing import Dict, Optional, Union
import pandas as pd
from chartalist.datasets.chartalist_dataset import ChartaListDataset


class EthereumTransactionNetworkDataset(ChartaListDataset):
    """
    Ethereum dataset.
    """

    _NOT_IN_DATASET: int = -1
    _data_frame = pd.DataFrame()
    _dataset_name: str = "ethereum"
    _versions_dict: Dict[str, Dict[str, Union[str, int]]] = {

        "trans_net_bancor": {
            "download_url": "https://chartalist.org/files/networkbancor.txt",
            "compressed_size": 4_841_472,
            "file_name": "networkbancor.txt",
            "labels": ["fromAddress ", "toAddress", "time", "amount"],
            "sep": " ",
        },
        "type_prediction_trans": {
            "download_url": "https://chartalist.org/data/ethTypePrediction/token_transfers_full.csv",
            "compressed_size": 2_300_300,
            "file_name": "token_transfers_full.csv",
            "labels": ["token_address", "from_address", "to_address", "value", "transaction_hash", "log_index",
                       "block_number"],
            "sep": ",",
        },
        "type_prediction_labels": {
            "download_url": "https://chartalist.org/data/ethTypePrediction/exchangeLabels.csv",
            "compressed_size": 170_300,
            "file_name": "exchangeLabels.csv",
            "labels": ["type", "address", "name"],
            "sep": ",",
        },
        "multilayer_bytom": {
            "download_url": "https://chartalist.org/data/ethMultilayerData/networkbytom.txt",
            "compressed_size": 458_472,
            "file_name": "networkbytom.txt",
            "labels": ["fromAddress ", "toAddress", "time", "amount"],
            "sep": " ",
        },
        "multilayer_cybermiles": {
            "download_url": "https://chartalist.org/data/ethMultilayerData/networkcybermiles.txt",
            "compressed_size": 314_480,
            "file_name": "networkcybermiles.txt",
            "labels": ["fromAddress ", "toAddress", "time", "amount"],
            "sep": " ",
        },
        "multilayer_decentraland": {
            "download_url": "https://chartalist.org/data/ethMultilayerData/networkdecentraland.txt",
            "compressed_size": 680_800,
            "file_name": "networkdecentraland.txt",
            "labels": ["fromAddress ", "toAddress", "time", "amount"],
            "sep": " ",
        },
        "multilayer_tierion": {
            "download_url": "https://chartalist.org/data/ethMultilayerData/networktierion.txt",
            "compressed_size": 397_670,
            "file_name": "networktierion.txt",
            "labels": ["fromAddress ", "toAddress", "time", "amount"],
            "sep": " ",
        },

        "multilayer_vechain": {
            "download_url": "https://chartalist.org/data/ethMultilayerData/networkvechain.txt",
            "compressed_size": 532_630,
            "file_name": "networkvechain.txt",
            "labels": ["fromAddress ", "toAddress", "time", "amount"],
            "sep": " ",
        },

        "multilayer_zrx": {
            "download_url": "https://chartalist.org/data/ethMultilayerData/networkzrx.txt",
            "compressed_size": 1_000_300,
            "file_name": "networkzrx.txt",
            "labels": ["fromAddress ", "toAddress", "time", "amount"],
            "sep": " ",
        },
        "stablecoin_erc20": {
            "download_url": "https://chartalist.org/data/stablecoinERC20/token_transfers.csv",
            "compressed_size": 1_700_300,
            "file_name": "token_transfers.csv",
            "labels": ["block_number", "transaction_index", "from_address", "to_address", "time_stamp",
                       "contract_address", "value"],
            "sep": ",",
        },

        "price_prediction_vechain": {
            "download_url": "https://chartalist.org/data/ethPricePrediction/networkvechainTX.txt",
            "compressed_size": 11_700_300,
            "file_name": "networkvechainTX.txt",
            "labels": ["fromAddress ", "toAddress", "time", "amount"],
            "sep": " ",
        },
        "price_prediction_zrx": {
            "download_url": "https://chartalist.org/data/ethPricePrediction/networkzrxTX.txt",
            "compressed_size": 13_260_300,
            "file_name": "networkzrxTX.txt",
            "labels": ["fromAddress ", "toAddress", "time", "amount"],
            "sep": " ",
        },
        "anomaly_detection_ether_delta_trades": {
            "download_url": "https://chartalist.org/data/ethAnomalyDetection/EtherDeltaTrades.csv",
            "compressed_size": 2_900_300,
            "file_name": "EtherDeltaTrades.csv",
            "labels": ["transaction_hash",	"block_number",	"timestamp",	"tokenGet",	"amountGet",	"tokenGive",	"amountGive",	"get",	"give"],
            "sep": ",",
        },
        "anomaly_detection_ether_dollar_price": {
            "download_url": "https://chartalist.org/data/ethAnomalyDetection/EtherDollarPrice.csv",
            "compressed_size": 60_300,
            "file_name": "EtherDollarPrice.csv",
            "labels": ["Date(UTC)", "UnixTimeStamp", "Value"],
            "sep": ",",
        },
        "anomaly_detection_idex": {
            "download_url": "https://chartalist.org/data/ethAnomalyDetection/IDEXTrades.csv",
            "compressed_size": 3_700_300,
            "file_name": "IDEXTrades.csv",
            "labels": ["transaction_hash", "status", "block_number", "gas", "gas_price", "timestamp", "amountBuy",
                       "amountSell", "expires", "nonce", "amount", "tradeNonce", "feeMake", "feeTake", "tokenBuy",
                       "tokenSell", "maker", "taker"],
            "sep": ",",
        },

    }

    def __init__(
            self,
            version: str = None,
            root_dir: str = "data",
            download: bool = False,
            split_scheme: str = "official",
    ):
        # Dataset information
        self._version: Optional[str] = version
        # The official split is to split by users
        self._split_scheme: str = "official"
        # Path of the dataset
        self._data_dir: str = self.initialize_data_dir(root_dir, download,
                                                       self._versions_dict[self.version]["file_name"])

        print("The Ethereum sample data downloaded successfully and stored on your local disk  -> {} \n"
              "---- For more information and downloading full datasets visit https://www.Chartalist.org ---- \n".format(
            self.version))
        # Load data
        data_df: pd.DataFrame = pd.read_csv(
            os.path.join(self.data_dir, self._versions_dict[self.version]["file_name"]),
            names=self._versions_dict[self.version]["labels"],
            keep_default_na=False,
            usecols=range(len(self._versions_dict[self.version]["labels"])),
            sep=self._versions_dict[self.version]["sep"],
            na_values=[],
            quoting=csv.QUOTE_NONNUMERIC,
        )
        self._data_frame = data_df
        super().__init__(root_dir, download, self._split_scheme)
