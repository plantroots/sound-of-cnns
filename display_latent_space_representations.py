import numpy as np
import random
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt

from vae import VAE
from utils import read_audio_files
from train import peak_amplitude_normalization, NUM_OF_SAMPLES_IN_A_FILE

MODEL_PATH = r"c:\Models\1024_FS_5LS"
AUDIO_FILES_PATH = r"c:\Dataset\filtered_kicks_small"

vae = VAE.load(MODEL_PATH)

NORMALIZED_SIGNALS = []
for signal, metadata in read_audio_files(AUDIO_FILES_PATH):

    normalized_signal = peak_amplitude_normalization(signal)

    if len(normalized_signal) < NUM_OF_SAMPLES_IN_A_FILE:
        padding_to_add = NUM_OF_SAMPLES_IN_A_FILE - len(normalized_signal)
        normalized_signal = np.append(normalized_signal, np.zeros(padding_to_add))

    normalized_signal = normalized_signal[np.newaxis, ...]
    normalized_signal = normalized_signal.reshape(40960, 1)
    NORMALIZED_SIGNALS.append(normalized_signal)

NORMALIZED_SIGNALS = np.array(NORMALIZED_SIGNALS)
# TODO: compile needs trainable=False and layers need to be set like this
for layer in vae.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

predictions = vae.predict(NORMALIZED_SIGNALS)
print(predictions)

# SAME INPUT -> DIFFERENT RESULTS PROBLEM
# TODO: check the predict method. the weights are the same
# ex_1:
# array([[-0.03399888, -0.04987655,  1.3686486 ,  3.9165041 ,  5.8359346 ],
#        [ 0.13973439, -0.25053236,  2.3458815 ,  2.5268898 ,  3.8385592 ]],
