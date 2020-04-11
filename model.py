import os
import random
import numpy as np
import pandas as pd
import librosa
import librosa.display
import seaborn as sb
from sklearn import model_selection
from sklearn import preprocessing
import matplotlib.pyplot as plt

# define directories
base_dir = "C:\\Users\\santa\\Desktop\\Python\\ambientVJ\\"
esc_dir = os.path.join(base_dir, "ESC-50-master\\")
meta_file = os.path.join(esc_dir, "meta\\esc50.csv")
audio_dir = os.path.join(esc_dir, "audio\\")


# load a wave data
def load_wave_data(audio_dir, file_name):
    file_path = os.path.join(audio_dir, file_name)
    x, fs = librosa.load(file_path, sr=44100)
    return x, fs


# change wave data to mel-stft
def calculate_melsp(x, n_fft=1024, hop_length=128):
    stft = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length)) ** 2
    log_stft = librosa.power_to_db(stft)
    melsp = librosa.feature.melspectrogram(S=log_stft, n_mels=128)
    return melsp


# display wave in plots
def show_wave(x):
    plt.plot(x)
    plt.show()


# display wave in heatmap
def show_melsp(melsp, fs):
    librosa.display.specshow(melsp, sr=fs)
    plt.colorbar()
    plt.show()


def main():
    print(esc_dir)
    print(meta_file)
    print(audio_dir)

    # load meta data
    meta_data = pd.read_csv(meta_file)

    # get data size
    data_size = meta_data.shape
    print(data_size)

    # arrange target label and its name
    class_dict = {}
    for i in range(data_size[0]):
        if meta_data.loc[i, "target"] not in class_dict.keys():
            class_dict[meta_data.loc[i, "target"]
                       ] = meta_data.loc[i, "category"]

    # example data
    x, fs = load_wave_data(audio_dir, meta_data.loc[0, "filename"])
    melsp = calculate_melsp(x)
    print("wave size:{0}\nmelsp size:{1}\nsamping rate:{2}".format(
        x.shape, melsp.shape, fs))
    show_wave(x)
    show_melsp(melsp, fs)
