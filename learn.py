import numpy as np
import os
import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation
from keras.layers import Conv2D, GlobalAveragePooling2D
from keras.layers import BatchNormalization, Add
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model

# dataset files
train_files = ["esc_melsp_train_raw.npz", "esc_melsp_train_ss.npz",
               "esc_melsp_train_st.npz", "esc_melsp_train_wn.npz", "esc_melsp_train_com.npz"]
test_file = "esc_melsp_test.npz"

freq = 128
time = 1723

train_num = 1500
test_num = 500


def cba(inputs, filters, kernel_size, strides):
    x = Conv2D(filters, kernel_size=kernel_size,
               strides=strides, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def main():
    x_train = np.zeros(freq*time*train_num*len(train_files)
                       ).reshape(train_num * len(train_files), freq, time)
    y_train = np.zeros(train_num * len(train_files))

    # redefine target data into one hot vector
    classes = 50
    # y_train = keras.utils.to_categorical(y_train, classes)
    # y_test = keras.utils.to_categorical(y_test, classes)

    # load dataset
    for i in range(len(train_files)):
        data = np.load(train_files[i])
        x_train[i * train_num: (i + 1) * train_num] = data["x"]
        y_train[i * train_num: (i + 1) * train_num] = data["y"]

    # load test dataset
    test_data = np.load(test_file)
    x_test = test_data["x"]
    y_test = test_data["y"]

    # reshape training dataset
    x_train = x_train.reshape(train_num * 5, freq, time, 1)
    x_test = x_test.reshape(test_num, freq, time, 1)

    y_test = keras.utils.to_categorical(y_test, classes)
    x_test = x_test.reshape(test_num, freq, time, 1)

    print("x train:{0}\ny train:{1}\nx test:{2}\ny test:{3}".format(x_train.shape,
                                                                    y_train.shape,
                                                                    x_test.shape,
                                                                    y_test.shape))

    # define CNN
    inputs = Input(shape=(x_train.shape[1:]))

    x_1 = cba(inputs, filters=32, kernel_size=(1, 8), strides=(1, 2))
    x_1 = cba(x_1, filters=32, kernel_size=(8, 1), strides=(2, 1))
    x_1 = cba(x_1, filters=64, kernel_size=(1, 8), strides=(1, 2))
    x_1 = cba(x_1, filters=64, kernel_size=(8, 1), strides=(2, 1))

    x_2 = cba(inputs, filters=32, kernel_size=(1, 16), strides=(1, 2))
    x_2 = cba(x_2, filters=32, kernel_size=(16, 1), strides=(2, 1))
    x_2 = cba(x_2, filters=64, kernel_size=(1, 16), strides=(1, 2))
    x_2 = cba(x_2, filters=64, kernel_size=(16, 1), strides=(2, 1))

    x_3 = cba(inputs, filters=32, kernel_size=(1, 32), strides=(1, 2))
    x_3 = cba(x_3, filters=32, kernel_size=(32, 1), strides=(2, 1))
    x_3 = cba(x_3, filters=64, kernel_size=(1, 32), strides=(1, 2))
    x_3 = cba(x_3, filters=64, kernel_size=(32, 1), strides=(2, 1))

    x_4 = cba(inputs, filters=32, kernel_size=(1, 64), strides=(1, 2))
    x_4 = cba(x_4, filters=32, kernel_size=(64, 1), strides=(2, 1))
    x_4 = cba(x_4, filters=64, kernel_size=(1, 64), strides=(1, 2))
    x_4 = cba(x_4, filters=64, kernel_size=(64, 1), strides=(2, 1))

    x = Add()([x_1, x_2, x_3, x_4])

    x = cba(x, filters=128, kernel_size=(1, 16), strides=(1, 2))
    x = cba(x, filters=128, kernel_size=(16, 1), strides=(2, 1))

    x = GlobalAveragePooling2D()(x)
    x = Dense(classes)(x)
    x = Activation("softmax")(x)

    model = Model(inputs, x)
    model.summary()
