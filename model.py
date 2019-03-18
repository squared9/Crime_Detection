import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

import datetime
import sys
import os

from tqdm import tqdm_notebook as tqdm

# uncomment for FP16 training
# Warning: on 2080Ti loss keeps at "inf" when using FP16
# from keras.backend.common import set_floatx
# set_floatx('float16')

import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Input
import tensorflow as tf
from keras.layers import Reshape, BatchNormalization
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed

from configuration import number_of_coordinates, number_of_frames

# TODO: extend with variational layers for better estimations of "unknown" positions
def get_autoencoder(input_dimension, sizes, activation="relu", is_variational=False, verbose=False):
    inputs = Input(shape=input_dimension, name="encoder_input")
    layer = inputs
    layer = BatchNormalization(name="normalization")(layer)  # test batch normalization for autoencoder
    for i, size in enumerate(sizes):
        layer = Dense(size, activation=activation, name="encoder_" + str(i))(layer)
    encoder = Model(inputs=inputs, outputs=layer)
    
    if verbose:
        encoder.summary()
        
    bottleneck_dimension = tuple(list(input_dimension[:-1]) + [sizes[-1]])
    
    encoded_inputs = Input(shape=bottleneck_dimension, name="decoder_input")
    layer = encoded_inputs
    for i, size in enumerate(reversed(sizes[:-1])):
        layer = Dense(size, activation=activation, name="decoder_" + str(i))(layer)
    outputs = Dense(input_dimension[1], activation="linear")(layer)
    decoder = Model(inputs=encoded_inputs, outputs=outputs)
    
    if verbose:
        decoder.summary()
        
    model = Model(inputs=encoder.inputs, outputs=decoder(encoder.outputs))
    
    if verbose:
        model.summary()
        
    return model, encoder, decoder


# test_autoencoder_only = True
test_autoencoder_only = False

def get_sequence_model(inner_model, frame_dimension, number_of_frames=30, lstm_activation="tanh", verbose=False):

    input_dimension = tuple([number_of_frames] + list(frame_dimension))
    inputs = Input(shape=input_dimension)

    # wrap an autoencoder inside a time-distributed layer for each time-step
    repeating_model = TimeDistributed(inner_model)(inputs)
    
    # reshape coordinates from (X, Y) to (X concat Y) for LSTM
    repeating_shape = repeating_model.get_shape().as_list()
    repeating_shape = tuple(repeating_shape[1:-2] + [repeating_shape[-2] * repeating_shape[-1]])
    
    sequence_layer = Reshape(repeating_shape)(repeating_model)
    
#     outputs = lstm_layer
    if test_autoencoder_only:
        outputs = Reshape(input_dimension)(sequence_layer)  # to debug autoencoder
    else:
        # put each autoencoder output to a LSTM with as many units as frames in a sequence
        lstm_layer = LSTM(units=2 * number_of_coordinates, activation=lstm_activation, return_sequences=True)(sequence_layer)
        lstm_layer = LSTM(units=2 * number_of_coordinates, activation=lstm_activation, return_sequences=True)(lstm_layer)
        lstm_layer = LSTM(units=2 * number_of_coordinates, activation=lstm_activation, return_sequences=True)(lstm_layer)
#         lstm_layer = LSTM(units=2 * number_of_coordinates, activation=lstm_activation, return_sequences=True)(lstm_layer)
#         lstm_layer = LSTM(units=2 * number_of_coordinates, return_sequences=True, activation="linear")(lstm_layer)
        outputs = Reshape(input_dimension)(lstm_layer)
        outputs = Dense(number_of_coordinates, activation="linear")(outputs)  # regression layer

    model = Model(inputs=inputs, outputs=outputs)
    if verbose:
        model.summary()

    return model
