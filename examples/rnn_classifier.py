import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, GRU
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
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


def LSTM_classifier(data, labels):
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
    y_pred_LSTM = model_LSTM.predict(data_test)
    y_pred_LSTM = np.where(y_pred_LSTM > 0.5, 1, 0)
    roc_LSTM = roc_auc_score(labels_test, y_pred_LSTM)

    # Evaluate the model on the test set
    loss, accuracy = model_LSTM.evaluate(data_test, labels_test)
    print(f"Loss: {loss}")
    print(f"Accuracy: {accuracy}")
    print("AUC: {}".format(roc_LSTM))


if __name__ == "__main__":
    # networkList = ["networkaeternity.txt", "networkaion.txt", "networkaragon.txt", "networkbancor.txt", "networkcentra.txt", "networkcindicator.txt", "networkcoindash.txt", "networkdgd.txt", "networkiconomi.txt"]
    networkList = ["networkadex.txt"]
    # tdaDifferentGraph = ["Overlap_0.1_Ncube_2", "Overlap_0.1_Ncube_5", "Overlap_0.2_Ncube_2", "Overlap_0.2_Ncube_5", "Overlap_0.3_Ncube_2", "Overlap_0.3_Ncube_5", "Overlap_0.5_Ncube_2", "Overlap_0.5_Ncube_5", "Overlap_0.6_Ncube_2", "Overlap_0.6_Ncube_5"]
    for network in networkList:
        # for tdaVariable in tdaDifferentGraph:
        print("Working on {}\n".format(network))
        data = read_seq_data(network)
        np_data = np.array(data["sequence"])
        np_labels = np.array(data["label"])
        LSTM_classifier(np_data, np_labels)

    # GCN_classifier(data)
