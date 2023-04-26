import csv
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from chartalist.datasets.chartalist_dataset import ChartaListDataset


class BitcoinTransactionNetworkDataset(ChartaListDataset):
    """
    Bitcoin dataset.
    Bitcoin Transaction Network Files
    """

    _NOT_IN_DATASET: int = -1

    _dataset_name: str = "bitcoin"
    _versions_dict: Dict[str, Dict[str, Union[str, int]]] = {
        "trans_net_in": {
            "download_url": "https://chartalist.org/data/73200_in.csv",
            "compressed_size": 73_724,
            "file_name": "73200_in.csv",
            "sep": ",",
            "labels": ["trans"],

        },
        "trans_net_out": {
            "download_url": "https://chartalist.org/data/73200_out.csv",
            "compressed_size": 38_042,
            "file_name": "73200_out.csv",
            "labels": ["trans"],
            "sep": ",",

        },
        "block_time": {
            "download_url": "https://chartalist.org/data/bitcoin_times.csv",
            "compressed_size": 13_384_999,
            "file_name": "bitcoin_times.csv",
            "labels": ["unix_time"],
            "sep": ",",

        },
        "price_prediction": {
            "download_url": "https://chartalist.org/data/pricedBitcoin2009-2018.csv",
            "compressed_size": 90_780,
            "file_name": "pricedBitcoin2009-2018.csv",
            "labels": ["date", "price", "year", "timeWindow", "totaltx"],
            "sep": ",",

        },
        "type_prediction": {
            "download_url": "https://chartalist.org/files/data_5.zip",
            "compressed_size": 113_384_999,
            "file_name": "BitcoinHeistData.csv",
            "labels": ["address", "year", "timeWindow", "length", "weight", "count", "looped", "neighbors", "income", "label"],
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
        self._data_dir: str = self.initialize_data_dir(root_dir, download,self._versions_dict[self.version]["file_name"] )

        print("The Bitcoin sample data downloaded successfully and stored on your local disk  -> {} \n"
              "---- For more information and downloading full datasets visit https://www.Chartalist.org ---- \n".format(
            self.version))
        # Load data
        data_df: pd.DataFrame = pd.read_csv(
            os.path.join(self.data_dir, self._versions_dict[self.version]["file_name"]),
            keep_default_na=False,
            names=self._versions_dict[self.version]["labels"],
            sep=self._versions_dict[self.version]["sep"],
            na_values=[],
            header=0,
            quoting=csv.QUOTE_NONNUMERIC,
        )
        self._data_frame = data_df
        super().__init__(root_dir, download, self._split_scheme)
