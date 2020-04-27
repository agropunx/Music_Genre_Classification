# Recurrent Neural Network classifier applied to music genre classification
# RNN : Long Short Term Memory
# output layer with 10 class/genres
# databse for training/testing see 3_Dataprep.py

import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# path to json file that stores MFCCs and genre labels for each processed segment
DATA_PATH = "./data.json"
"""
config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.9
keras.backend.set_session(tf.Session(config=config));"""


def load_data(data_path):
    """Loads training dataset from json file.
        :param data_path (str): Path to json file containing data
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """

    with open(data_path, "r") as fp:
        data = json.load(fp)

    # convert lists to numpy arrays
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    print("Data succesfully loaded!")

    return X, y

def prepare_datasets(test_size, validation_size):

    # load data
    X, y = load_data(DATA_PATH)

    # create the train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # create train/validation spit
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)


    return X_train, X_validation, X_test, y_train, y_validation, y_test

def build_model(input_shape):
#Build an RNN-LSTM model

    #create model
    model = keras.Sequential()

    #create 2 LSTM layers
    model.add(keras.layers.LSTM(64,input_shape=input_shape, return_sequences=True ))  # this is sequence to sequence
    model.add(keras.layers.LSTM(64))  #64 unit per layer # this is sequence to vector!

    # create a dense layer
    model.add(keras.layers.Dense(64,activation ="relu"))
    model.add(keras.layers.Dropout(0.3))

    #softmax output layer
    model.add(keras.layers.Dense( 10, activation = 'softmax')) # each sample is classified with a probability for each of the 10 genres

    return model

def plot_history(history):
    """Plots accuracy/loss for training/validation set as a function of the epochs
        :param history: Training history of model
        :return:
    """

    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["acc"], label="train accuracy")
    axs[0].plot(history.history["val_acc"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()

def predict(model,X,y):

    X= X[np.newaxis, ...]      # X is a 3d array at this stage (130,13,1) , but the model expect a 4d array (1,130,13,1)
    predictions = model.predict(X)  #prediction is a 2d array [[ p1 ,p2, p3,...,p10]] 10 values/genre

    #extract index with max value
    predicted_index = np.argmax(predictions, axis=1) #1d array with a number in range 0-9
    #create index list from json file
    index_list= [
        "jazz",
        "reggae",
        "metal",
        "blues",
        "rock",
        "pop",
        "hiphop",
        "classical",
        "country",
        "disco"
    ]
    print("Expected genre is equal to {} and the predicted genre is equal to {}".format(index_list[int(y)], index_list[int(predicted_index)]))

if __name__ == "__main__":

    # 1 create train, validation and test sets
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25 , 0.2)

    # 2 build the CNN net
    input_shape = (X_train.shape[1],X_train.shape[2]) # 2d array
    model = build_model(input_shape)

    # 3 compile the network
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)

    model.compile(optimizer, loss="sparse_categorical_crossentropy", metrics = ['accuracy'] )

    model.summary()

    #conf for gpu usage
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    # 4 train the CNN
    history = model.fit(X_train, y_train, validation_data = (X_validation,y_validation),batch_size = 32, epochs = 30 )

    # 5 evaluate CNN on the test set
    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose = 2 )
    print("Accuracy on test set is:{}".format(test_accuracy))

    plot_history(history)

    # 6 make prediction on a sample
    X = X_test[100]
    y = y_test[100]
    predict(model,X,y)
