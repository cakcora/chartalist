import csv
import os
import pickle
import re
import time
import pandas as pd
import datetime as dt
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, GRU
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
# import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import LSTM, Dense


def read_seq_data(network):
    file_path = "Sequence/{}/".format(network)
    file = "seq.txt"
    seqData = dict()
    with open(file_path + file, 'rb') as f:
        # print("\n Reading Torch Data {} / {}".format(inx, len(files)))
        seqData = pickle.load(f)
    return seqData


def read_seq_data_by_file_name(network, file):
    file_path = "Sequence/{}/".format(network)
    seqData = dict()
    with open(file_path + file, 'rb') as f:
        # print("\n Reading Torch Data {} / {}".format(inx, len(files)))
        seqData = pickle.load(f)
    return seqData


def LSTM_classifier(data, labels, spec, network):
    start_Rnn_training_time = time.time()
    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Define the LSTM model
    model_LSTM = Sequential()
    model_LSTM.add(LSTM(64, input_shape=(7, 3), return_sequences=True))  # Adjust the number of units as needed
    # model_LSTM.add(LSTM(64, activation='relu', input_shape=(45, 1), return_sequences=True))
    model_LSTM.add(LSTM(32, activation='relu', return_sequences=True))
    model_LSTM.add(GRU(32, activation='relu', return_sequences=True))
    model_LSTM.add(GRU(32, activation='relu', return_sequences=False))
    model_LSTM.add((Dense(100, activation='relu')))

    model_LSTM.add(Dense(1, activation="sigmoid"))

    # Compile the model
    model_LSTM.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    model_LSTM.fit(data_train, labels_train, epochs=100, batch_size=32,
                   validation_data=(data_test, labels_test))  # Adjust the epochs and batch_size as needed

    # Make predictions on the test set

    start_Rnn_training_time = time.time() - start_Rnn_training_time
    y_pred_LSTM = model_LSTM.predict(data_test)
    y_pred_LSTM = np.where(y_pred_LSTM > 0.5, 1, 0)
    roc_LSTM = roc_auc_score(labels_test, y_pred_LSTM)

    # Evaluate the model on the test set
    loss, accuracy = model_LSTM.evaluate(data_test, labels_test)
    print(f"Loss: {loss}")
    print(f"Accuracy: {accuracy}")
    print("AUC: {}".format(roc_LSTM))
    try:
        # Attempt to open the file in 'append' mode
        with open("RnnResults/RNN-Results.txt", 'a') as file:
            # Append a line to the existing file
            file.write(
                "{},{},{},{},{},{},{}".format(network, spec, loss, accuracy, roc_LSTM, start_Rnn_training_time,
                                              len(data)) + '\n')
    except FileNotFoundError:
        # File doesn't exist, so create a new file and write text
        with open("RnnResults/RNN-Results.txt", 'w') as file:
            file.write(
                "Network={} Spec={} Loss={} Accuracy={} AUC={} time={} data={}".format(network, spec, loss, accuracy,
                                                                                       roc_LSTM,
                                                                                       start_Rnn_training_time,
                                                                                       len(data)) + '\n')


def merge_dicts(list_of_dicts):
    merged_dict = {}
    for dictionary in list_of_dicts:
        for key, value in dictionary.items():
            if key in merged_dict:
                merged_dict[key].append(value)
            else:
                merged_dict[key] = [value]
    return merged_dict


def outputCleaner():
    input_file = "RnnResults/AternityTest.txt"
    output_file = "RnnResults/Aternity_RNN_Results_cleaned_v2.csv"

    with open(input_file, "r") as file:
        lines = file.readlines()

    data = []
    for line in lines:
        values = line.strip().split()
        spec_values = values[1].split("-")
        row = [value.split("=")[1] for value in values[:1]] + spec_values + [value.split("=")[1] for value in
                                                                             values[2:]]
        data.append(row)

    with open(output_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(data)
        file.close()

    with open(output_file, "r") as file:
        reader = csv.reader(file)
        for row in reader:
            new_row = []
            for i, value in enumerate(row):
                if i in (1, 2, 3):
                    numeric_part = re.findall(r"[-+]?\d*\.?\d+", value)
                    if numeric_part:
                        new_row.append(numeric_part[0])
                    else:
                        new_row.append("")
                else:
                    new_row.append(value)
            data.append(new_row)

    with open(output_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(data)

    print("Conversion complete. CSV file created.")


def visualize_time_exp():
    # Specify the file path of the CSV
    file_path = "TDA_time_exp_RNN.csv"

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)
    x = df["Snap"]
    y = df["Time"]
    networkName = df["Net"]

    # Calculate the regression line
    regression_line = np.polyfit(x, y, 1)
    regression_y = np.polyval(regression_line, x)

    # Add labels to each point
    for i in range(len(x)):
        plt.text(x[i], y[i] - 0.5, f'{networkName[i]}', ha='center', va='bottom', fontsize=7)

    # Create the scatter plot
    plt.plot(x, y, label='Transaction Networks', marker='o')
    # plt.plot(x, regression_y, color='red', label='Regression Line')

    # Set the y-axis to logarithmic scale
    # plt.yscale('log')

    # Set axis labels and title
    plt.xlabel('# Snapshots')
    plt.ylabel('Time (Sec)')
    plt.title('RNN training costs of token networks')

    # Display the plot
    plt.legend()
    plt.tight_layout()
    plt.savefig('tda_time_exp_Rnn_plot.png', dpi=300)
    plt.show()

    # Display the DataFrame
    print(df)


def visualize_time_exp_bar():
    # Specify the file path of the CSV
    file_path = "TDA_time_exp_RNN.csv"

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)
    x = df["Net"]
    y = df["Time"]
    plt.bar(x, y)

    # # Adding labels to each bar
    # for i in range(len(x)):
    #     plt.text(i, y[i], str(y[i]), ha='center', va='bottom')
    #
    # # Calculating trend line
    # z = np.polyfit(range(len(x)), y, 1)
    # p = np.poly1d(z)
    # plt.plot(range(len(x)), p(range(len(x))), 'r--')
    #
    # # Adding trend line equation as a text
    # plt.text(len(x) - 1, p(len(x) - 1), f"Trend: {z[0]:.2f}x + {z[1]:.2f}", ha='right', va='center',
    #          color='red')

    # Customizing the plot
    plt.xlabel('Networks')
    plt.ylabel('Time (sec)')
    plt.title('RNN training cost')

    # Displaying the plot
    plt.savefig('tda_time_exp_Rnn_bar.png', dpi=300)
    plt.show()


def visualize_time_exp_scatter():
    # Specify the file path of the CSV
    file_path = "TDA_time_exp.csv"

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)
    x = df["Node"]
    y = df["Time"]

    # Calculate the regression line
    regression_line = np.polyfit(x, y, 1)
    regression_y = np.polyval(regression_line, x)

    # Add labels to each point

    # Create the scatter plot
    plt.scatter(x, y, label='Transaction Networks', s=6)
    plt.plot(x, regression_y, color='red', label='Regression Line')

    # Set the y-axis to logarithmic scale
    plt.yscale('log')

    # Set axis labels and title
    plt.xlabel('# Nodes')
    plt.ylabel('Time (Sec)')
    plt.title('TDA costs of daily token networks')

    # Display the plot
    plt.legend()
    plt.tight_layout()
    plt.savefig('tda_time_exp_plot_r.png', dpi=300)
    plt.show()

    # Display the DataFrame
    print(df)


def getDailyAvg(file):
    timeseries_file_path = "../data/all_network/TimeSeries/"
    timeseries_file_path_other = "../data/all_network/TimeSeries/Other/"
    selectedNetwork = pd.read_csv((timeseries_file_path_other + file), sep=' ', names=["from", "to", "date"])
    selectedNetwork['value'] = 1
    selectedNetwork['date'] = pd.to_datetime(selectedNetwork['date'], unit='s').dt.date
    selectedNetwork['value'] = selectedNetwork['value'].astype(float)
    selectedNetwork = selectedNetwork.sort_values(by='date')
    window_start_date = selectedNetwork['date'].min()
    data_last_date = selectedNetwork['date'].max()
    days_of_data = (data_last_date - window_start_date).days
    avg_daily_trans = len(selectedNetwork) / days_of_data
    # Concatenate the two columns
    combined = pd.concat([selectedNetwork['from'], selectedNetwork['to']], ignore_index=True)
    # Get the number of unique items
    num_unique = combined.nunique()
    avg_daily_nodes = num_unique / days_of_data
    print(
        f"AVG daily stat for {file} -> nodes = {avg_daily_nodes} , edges = {avg_daily_trans} , days = {days_of_data} , total trans = {len(selectedNetwork)}")


def getDailyAvgReddit(file):
    timeseries_file_path = "../data/all_network/TimeSeries/"
    timeseries_file_path_other = "../data/all_network/TimeSeries/Other/"

    selectedNetwork = pd.read_csv((timeseries_file_path_other + file), sep='\t')
    selectedNetwork = selectedNetwork[["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT", "TIMESTAMP", "LINK_SENTIMENT"]]
    column_mapping = {
        'SOURCE_SUBREDDIT': 'from',
        'TARGET_SUBREDDIT': 'to',
        'TIMESTAMP': 'date',
        'LINK_SENTIMENT': 'value'
    }
    selectedNetwork.rename(columns=column_mapping, inplace=True)
    selectedNetwork['date'] = pd.to_datetime(selectedNetwork['date']).dt.date
    selectedNetwork['value'] = selectedNetwork['value'].astype(float)
    selectedNetwork = selectedNetwork.sort_values(by='date')
    # reddit 800
    window_start_date = selectedNetwork['date'].min() + dt.timedelta(days=800)
    data_last_date = selectedNetwork['date'].max()
    days_of_data = (data_last_date - window_start_date).days
    avg_daily_trans = len(selectedNetwork) / days_of_data
    # Concatenate the two columns
    combined = pd.concat([selectedNetwork['from'], selectedNetwork['to']], ignore_index=True)
    # Get the number of unique items
    num_unique = combined.nunique()
    avg_daily_nodes = num_unique / days_of_data
    print(
        f"AVG daily stat for {file} -> nodes = {avg_daily_nodes} , edges = {avg_daily_trans} , days = {days_of_data} , total trans = {len(selectedNetwork)}, Range {window_start_date} {data_last_date}")

def write_list_to_csv(filename, data_list):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        for item in data_list:
            writer.writerow([item])

if __name__ == "__main__":
    # for generating daily stats
    # processingIndx = 0
    # timeseries_file_path_other = "../data/all_network/TimeSeries/Other/"
    # files = os.listdir(timeseries_file_path_other)
    # for file in files:
    #     if file.endswith(".txt"):
    #         print("Processing {} / {} \n".format(processingIndx, len(files) - 4))
    #         getDailyAvg(file)
    #         processingIndx += 1

    # visualize_time_exp_bar()
    # #outputCleaner()
    # # "networkaeternity.txt", "networkaion.txt", "networkaragon.txt", "networkbancor.txt", "networkcentra.txt", "networkcindicator.txt", "networkcoindash.txt" , "networkiconomi.txt", "networkadex.txt"
    # # "networkdgd.txt","networkcentra.txt","networkcindicator.txt"
    networkList = ["Reddit_B.tsv"]
    # networkList = ["mathoverflow.txt", "networkcoindash.txt", "networkiconomi.txt", "networkadex.txt", "networkdgd.txt",
    #                "networkbancor.txt", "networkcentra.txt", "networkcindicator.txt", "networkaeternity.txt",
    #                "networkaion.txt", "networkaragon.txt", "CollegeMsg.txt", "Reddit_B.tsv"]
    # # tdaDifferentGraph = ["Overlap_0.1_Ncube_2", "Overlap_0.1_Ncube_5", "Overlap_0.2_Ncube_2", "Overlap_0.2_Ncube_5", "Overlap_0.3_Ncube_2", "Overlap_0.3_Ncube_5", "Overlap_0.5_Ncube_2", "Overlap_0.5_Ncube_5", "Overlap_0.6_Ncube_2", "Overlap_0.6_Ncube_5"]
    for network in networkList:
        # for tdaVariable in tdaDifferentGraph:
        print("Working on {}\n".format(network))
        data2 = read_seq_data_by_file_name(network, "seq.txt")
        labels = data2["label"]
        write_list_to_csv(network.split(".")[0]+"_Label.csv", labels)
        print(labels)

    #
    #
    #
    #     files = os.listdir(f"Sequence/{network}/dailySeq/")
    #     data = []
    #     result = {}
    #     for file in files:
    #         if file.endswith(".txt"):
    #             data.append(read_seq_data_by_file_name(network,  f"dailySeq/{file}"))
    #
    #
    #
    #     data = merge_dicts(data)
    #     for dictionary in data["sequence"]:
    #         for key, value in dictionary.items():
    #             if key not in result:
    #                 result[key] = []
    #             result[key].append(value)
    #
    #     data["sequence"] = result
    #
    #
    #     for key, value in data["sequence"].items():
    #         print("Processing network ({}) - with parameters {}".format(network, key))
    #         np_labels = np.array(data["label"])
    #         # if (len(value[0]) != 7):
    #         #     while (len(value[0]) != 7):
    #         #         del value[0]
    #         #         np_labels = np.delete(np_labels, 0, axis=0)
    #         indxs = []
    #         if (network == "networkaion.txt"):
    #             for i in range(0, len(value)):
    #                 if len(value[i]) != 7:
    #                    indxs.append(i)
    #
    #             value = [item for index, item in enumerate(value) if index not in indxs]
    #             np_labels =  np.delete(np_labels, indxs)
    #
    #
    #         np_data = np.array(value)
    #
    #
    #         LSTM_classifier(np_data, np_labels, key, network)

    # for seq files
    # for network in networkList:
    #     # for tdaVariable in tdaDifferentGraph:
    #     print("Working on {}\n".format(network))
    #     data = read_seq_data_by_file_name(network, "seq_raw.txt")
    #
    #     # indx = 0
    #     for key, value in data["sequence"].items():
    #         # if (indx == 1):
    #         #     break
    #         print("Processing network ({}) - with parameters {}".format(network, key))
    #
    #         np_labels = np.array(data["label"])
    #         if (len(value[0]) != 7):
    #             while (len(value[0]) != 7):
    #                 del value[0]
    #                 np_labels = np.delete(np_labels, 0, axis=0)
    #         indxs = []
    #         if (network == "networkdgd.txt"):
    #             for i in range(0, len(value)):
    #                 if len(value[i]) != 7:
    #                     indxs.append(i)
    #
    #             value = [item for index, item in enumerate(value) if index not in indxs]
    #             np_labels = np.delete(np_labels, indxs)
    #
    #         np_data = np.array(value)
    #         LSTM_classifier(np_data, np_labels, key, network)
            # indx += 1
    #
