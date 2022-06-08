from .version import __version__
from .get_dataset import get_dataset

benchmark_datasets = [
    'bitcoin',
    'ethereum',
    'dashcoin'

]

additional_datasets = [
]

supported_datasets = benchmark_datasets + additional_datasets

bitcoin_loaders = [
    'trans_net_out',
    'trans_net_in',
    'price_prediction',
    'type_prediction',
    'block_time'


]
ethereum_loaders = [
    'trans_net_bancor',
    'multilayer_bytom',
    'multilayer_cybermiles',
    'multilayer_decentraland',
    'multilayer_tierion',
    'multilayer_vechain',
    'multilayer_zrx',
    'stablecoin_erc20',
    'type_prediction_trans',
    'type_prediction_labels',
    'price_prediction_vechain',
    'price_prediction_zrx',
    'anomaly_detection_ether_delta_trades',
    'anomaly_detection_ether_dollar_price',
    'anomaly_detection_idex'




]
dashcoin_loaders = [
    'trans_net_out',
    'trans_net_in',
]

class BitcoinLoaders():
    TRANSACTION_NETWORK_INPUT_SAMPLE = 'trans_net_in'
    TRANSACTION_NETWORK_OUTPUT_SAMPLE = 'trans_net_out'
    PRICE_PREDICTION = 'price_prediction'
    TYPE_PREDICTION = 'type_prediction'
    BLOCK_TIME = 'block_time'

class EthereumLoader():
    TRANSACTION_NETWORK_BANCOR = 'trans_net_bancor'
    MULTILAYER_BYTOM = 'multilayer_bytom'
    MULTILAYER_CYBERMILES = 'multilayer_cybermiles'
    MULTILAYER_DECENTRALAND = 'multilayer_decentraland'
    MULTILAYER_TIERION = 'multilayer_tierion'
    MULTILAYER_VECHAIN = 'multilayer_vechain'
    MULTILAYER_ZRX = 'multilayer_zrx'
    STABLECOIN_ERC20 = 'stablecoin_erc20'
    TYPE_PREDICTION_TRANSACTIONS = 'type_prediction_trans'
    TYPE_PREDICTION_LABELS = 'type_prediction_labels'
    PRICE_PREDICTION_VECHAIN = 'price_prediction_vechain'
    PRICE_PREDICTION_ZRX = 'price_prediction_zrx'
    ANOMALY_DETECTION_ETHER_DELTA_TRADES = 'anomaly_detection_ether_delta_trades'
    ANOMALY_DETECTION_ETHER_DOLLAR_PRICE = 'anomaly_detection_ether_dollar_price'
    ANOMALY_DETECTION_IDEX = 'anomaly_detection_idex'

class DashcoinLoader():
     TRANSACTION_NETWORK_INPUT_SAMPLE  = 'trans_net_in'
     TRANSACTION_NETWORK_OUTPUT_SAMPLE = 'trans_net_out'


def get_info():
    print("For more information visit our website at https://www.chartalist.org \n")
    info = {
        "name": "Chartalist",
        "author": "Chartalist team",
        "Main Creator" : "Kiarash Shamsi",
        "author_email": "info@chartalist.org",
        "url": "https://Chartalist.org",
        "description": "Chartalist ML-Ready Dataset",
    }
    print(info)
    return info