from keras import Sequential
from keras.layers import LSTM, Conv1D, Dense, Dropout, Flatten, Input, MaxPooling1D

from src.models.attention import Attention
from src.models.dain import Dain


def generate_cnn(N=15):
    cnn = Sequential()

    cnn.add(Input(shape=(N, 2)))
    cnn.add(Dain(N, 2))
    cnn.add(Conv1D(filters=128, kernel_size=3, activation="relu", padding="same"))
    cnn.add(Conv1D(filters=64, kernel_size=3, activation="relu", padding="same"))
    cnn.add(Dropout(0.2))
    cnn.add(Flatten())
    cnn.add(Dense(128, activation="relu"))
    cnn.add(Dropout(0.2))
    cnn.add(Dense(64, activation="relu"))
    cnn.add(Dropout(0.2))
    cnn.add(Dense(1, activation="sigmoid"))

    cnn.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return cnn


def generate_lstm(N=15):
    lstm = Sequential()

    lstm.add(Input(shape=(N, 2)))
    lstm.add(Dain(N, 2))
    lstm.add(LSTM(units=128, return_sequences=True))
    lstm.add(Dropout(0.3))
    lstm.add(LSTM(units=64))
    lstm.add(Dropout(0.3))
    lstm.add(Flatten())
    lstm.add(Dense(128, activation="relu"))
    lstm.add(Dropout(0.2))
    lstm.add(Dense(64, activation="relu"))
    lstm.add(Dropout(0.2))
    lstm.add(Dense(1, activation="sigmoid"))

    lstm.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return lstm


def generate_cnn_lstm(N=15):
    cnn_lstm = Sequential()

    cnn_lstm.add(Input(shape=(N, 2)))
    cnn_lstm.add(Dain(N, 2))
    cnn_lstm.add(Conv1D(filters=128, kernel_size=3, activation="relu", padding="same"))
    cnn_lstm.add(Conv1D(filters=64, kernel_size=3, activation="relu", padding="same"))
    cnn_lstm.add(LSTM(units=128, return_sequences=True))
    cnn_lstm.add(Dropout(0.3))
    cnn_lstm.add(LSTM(units=64))
    cnn_lstm.add(Dropout(0.3))
    cnn_lstm.add(Flatten())
    cnn_lstm.add(Dense(128, activation="relu"))
    cnn_lstm.add(Dropout(0.2))
    cnn_lstm.add(Dense(64, activation="relu"))
    cnn_lstm.add(Dropout(0.2))
    cnn_lstm.add(Dense(1, activation="sigmoid"))

    cnn_lstm.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return cnn_lstm


def generate_cnn_lstm_attention(N=15):
    cnn_lstm_model = Sequential()

    cnn_lstm_model.add(Input(shape=(N, 2)))
    cnn_lstm_model.add(Dain(N, 2))
    cnn_lstm_model.add(
        Conv1D(filters=128, kernel_size=3, activation="relu", padding="same")
    )
    cnn_lstm_model.add(
        Conv1D(filters=64, kernel_size=3, activation="relu", padding="same")
    )
    cnn_lstm_model.add(LSTM(units=128, return_sequences=True))
    cnn_lstm_model.add(Dropout(0.3))
    cnn_lstm_model.add(LSTM(units=64, return_sequences=True))
    cnn_lstm_model.add(Dropout(0.3))
    cnn_lstm_model.add(Attention(64))
    cnn_lstm_model.add(Dropout(0.2))
    cnn_lstm_model.add(Flatten())
    cnn_lstm_model.add(Dense(128, activation="relu"))
    cnn_lstm_model.add(Dropout(0.2))
    cnn_lstm_model.add(Dense(64, activation="relu"))
    cnn_lstm_model.add(Dropout(0.2))
    cnn_lstm_model.add(Dense(1, activation="sigmoid"))

    cnn_lstm_model.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
    )
    return cnn_lstm_model
