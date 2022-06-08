import csv
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from chartalist.datasets.chartalist_dataset import ChartaListDataset


class DashcoinTransactionNetworkDataset(ChartaListDataset):
    """
    Dashcoin dataset.
    Dashcoin Transaction Network Files
    """

    _NOT_IN_DATASET: int = -1

    _dataset_name: str = "dashcoin"
    _versions_dict: Dict[str, Dict[str, Union[str, int]]] = {
        "trans_net_in": {
            "download_url": "https://chartalist.org/data/dash/6000_in.csv",
            "compressed_size": 29_424,
            "file_name": "6000_in.csv",
            "sep": ",",
            "labels": ["trans"],

        },
        "trans_net_out": {
            "download_url": "https://chartalist.org/data/dash/6000_out.csv",
            "compressed_size": 47_442,
            "file_name": "6000_out.csv",
            "labels": ["trans"],
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

        print("The Dashcoin sample data downloaded successfully and stored on your local disk  -> {} \n"
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
