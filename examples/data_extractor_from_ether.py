import pandas as pd
from google.cloud import bigquery



service_account_key = "cred4.json"
project_id = 'eng-spot-392514'
dataset_id = "`bigquery-public-data.crypto_ethereum.token_transfers`"
query = f"SELECT token_address, from_address, to_address, value, block_timestamp FROM {dataset_id} WHERE block_timestamp >= TIMESTAMP(\"2016-04-01\") AND block_timestamp < TIMESTAMP(\"2018-06-01\")"

client = bigquery.Client.from_service_account_json(service_account_key)
df = client.query(query).to_dataframe()

output_file = "path_to_output_file.csv"  # Replace with your desired file path and name
df.to_csv(output_file, index=False)


#
#
#
# import pandas
#
#
#
# file_name = 'dataset.csv'
#
# df = pandas.io.gbq.read_gbq(
# '''
# SELECT token_address, from_address, to_address, value, block_timestamp
# FROM `bigquery-public-data.crypto_ethereum.token_transfers`
# WHERE block_timestamp >= TIMESTAMP("2016-04-01")
# AND block_timestamp < TIMESTAMP("2018-06-01")
# LIMIT 1000
# ''',
# project_id = project_id, dialect = 'standard'
# )
#
# df.to_csv(file_name, index=False)