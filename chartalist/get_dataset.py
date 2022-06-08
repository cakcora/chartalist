from typing import Optional

import chartalist


def get_dataset(dataset: str, version: Optional[str] = None, data_frame: bool = False,
                **dataset_kwargs):
    """
    Returns the appropriate ChartaList dataset class.
    Input:
        dataset (str): Name of the dataset
        version (Union[str, None]): Dataset version number, e.g., '1.0'.
                                    Defaults to the latest version.
        dataset_kwargs: Other keyword arguments to pass to the dataset constructors.
    Output:
        The specified ChartaListDataset class.
    """
    if version is not None:
        version = str(version)

    if dataset not in chartalist.supported_datasets:
        raise ValueError(f'The dataset {dataset} is not recognized. Must be one of {chartalist.supported_datasets}.')

    if dataset == 'bitcoin':
        if version not in chartalist.bitcoin_loaders :
            raise ValueError(
                f'The Bitcoin loader version \" {version} \" is not recognized. Must be one of {chartalist.bitcoin_loaders}.')
        from chartalist.datasets.bitcoin_transaction_network_dataset import BitcoinTransactionNetworkDataset
        if data_frame:
            return BitcoinTransactionNetworkDataset(version=version, **dataset_kwargs).data_frame
        else:
            return BitcoinTransactionNetworkDataset(version=version, **dataset_kwargs)

    elif dataset == 'ethereum':
        if version not in chartalist.ethereum_loaders:
            raise ValueError(
                f'The Ethereum loader version \" {version} \" is not recognized. Must be one of {chartalist.ethereum_loaders}.')
        from chartalist.datasets.ethereum_transaction_network_dataset import EthereumTransactionNetworkDataset
        if data_frame:
            return EthereumTransactionNetworkDataset(version=version, **dataset_kwargs).data_frame
        else:
            return  EthereumTransactionNetworkDataset(version=version, **dataset_kwargs)

    elif dataset == 'dashcoin':
        if version not in chartalist.dashcoin_loaders:
            raise ValueError(
                f'The Dashcoin loader version \" {version} \" is not recognized. Must be one of {chartalist.dashcoin_loaders}.')
        from chartalist.datasets.dash_transaction_network_dataset import DashcoinTransactionNetworkDataset
        if data_frame:
            return DashcoinTransactionNetworkDataset(version=version, **dataset_kwargs).data_frame
        else:
            return DashcoinTransactionNetworkDataset(version=version, **dataset_kwargs)