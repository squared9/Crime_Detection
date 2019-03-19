import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

import datetime
import sys
import os

from tqdm import tqdm
from model import get_autoencoder, get_sequence_model

from configuration import (IDEAL_HUMAN_BODY_RATIO, NORMALIZE_ASPECT_RATIO, 
                           number_of_frames, base_number_of_coordinates, number_of_coordinates)
from generator_pair import Dual_Track_Generator
from loss import (pair_loss, cumulative_point_distance_error, mean_point_distance_error, 
                  pair_cumulative_point_distance_error, pair_mean_point_distance_error, 
                  loss_weight_adjustments, base_loss_function)

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.optimizers import Adam

# input shape of 2 poses/frame
input_shape = (2, 2 * number_of_coordinates)

# layer_sizes = [30, 20, 10]
# layer_sizes = [128, 64, 10]
# layer_sizes = [256, 128, 64, 10]
layer_sizes = [512, 256, 128, 64, 10]
autoencoder, encoder, decoder = get_autoencoder(input_shape, layer_sizes, is_variational=False, verbose=True)
model = get_sequence_model(autoencoder, (input_shape), number_of_frames=number_of_frames, factor=2, verbose=True)

sequences_df = pd.read_csv("pose_pair_sequences.csv", sep=",", header=0, index_col=None)
print(len(sequences_df), "records:")
print(sequences_df.head(5))

# get all counts
print()
print("Counts:")
counts_df = sequences_df.groupby(["sequence_id"], as_index=False).count().loc[:, ["sequence_id", "step"]]
print(counts_df.head(5))

# extract all sequence IDs with at least ${number_of_frames} steps
print()
print("Suitable training sequences:")
training_sequences_df = counts_df.loc[counts_df["step"] >= number_of_frames].loc[:, ["sequence_id"]]
print(training_sequences_df.head(5))

print()
print("Found", len(training_sequences_df), "suitable sequences for training.")

# join selected sequences and original dataset
training_df = pd.merge(training_sequences_df, sequences_df, how="inner", on=["sequence_id"])
training_df.sort_values(["sequence_id", "step"], ascending=True, inplace=True)
training_df.head(10)

print(len(training_df), "training records")

# x_a0..x_a16,y_a0..y_a16,x_b0..x_b16,y_b0..y_b16
coordinate_columns = ['x_a', 'y_a', 'x_b', 'y_b']
coordinate_columns = [x + str(i) for x in coordinate_columns for i in range(base_number_of_coordinates)]

print()
print("Pose columns:")
print(coordinate_columns)
print()

## Data preparation

def get_aspect_ratio_correction(width, height):
    """
    Returns aspect ratio correction need to reach ideal human body ratio
    """
    ratio = IDEAL_HUMAN_BODY_RATIO * width / height  # correction of human body ratio
    return ratio
    

# dictionary containing a list of tuples of a list of coordinates and a bounding box for each step
training_sequences = {}

# bounding box of the whole sequence for repositioning/augmentation
sequence_boundaries = {}

# maximal dimensions of individual tracks
track_dimensions = {}

LIMIT = 1E10

total = 0

print("Preparing training sequences")
progress_bar = tqdm(total=len(training_df))

previous_sequence_id = training_df.iloc[0]["sequence_id"]

min_x = min_y = min_xa = min_ya = min_xb = min_yb = LIMIT
max_x = max_y = max_xa = max_ya = max_xb = max_yb = -LIMIT
max_width = max_height = max_width_a = max_height_a = max_width_b = max_height_b = -LIMIT

MAX_ROWS = -1    # unlimited
# MAX_ROWS = 1024  # restrict for test
# MAX_ROWS = 10 * 1024 # restrict for test

for i, record in training_df.iterrows():
    if MAX_ROWS is not None and MAX_ROWS >= 0 and i > MAX_ROWS:
        break
    coordinates = record[coordinate_columns].values
#     print(coordinates)
#     print(coordinates.shape)

    min_xa = np.min(coordinates[:base_number_of_coordinates])
    min_xb = np.min(coordinates[2 * base_number_of_coordinates: 3 * base_number_of_coordinates])
    min_x = min(min_xa, min_xb)

    min_ya = np.min(coordinates[base_number_of_coordinates: 2 * base_number_of_coordinates])
    min_yb = np.min(coordinates[3 * base_number_of_coordinates: 4 * base_number_of_coordinates])
    min_y = min(min_ya, min_yb)

    max_xa = np.max(coordinates[:base_number_of_coordinates])
    max_xb = np.max(coordinates[2 * base_number_of_coordinates: 3 * base_number_of_coordinates])
    max_x = max(max_xa, max_xb)

    max_ya = np.max(coordinates[base_number_of_coordinates: 2 * base_number_of_coordinates])
    max_yb = np.max(coordinates[3 * base_number_of_coordinates: 4 * base_number_of_coordinates])
    max_y = max(max_ya, max_yb)
    
    bounding_box = [min_x,  # min x, top left
                    min_y,  # min y
                    max_x,  # max x, bottom right
                    max_y]  # max y
#     print(bounding_box)

    bounding_box_a = [min_xa,
                      min_ya,
                      max_xa,
                      max_ya]
#     print(bounding_box_a)

    bounding_box_b = [min_xb,
                      min_yb,
                      max_xb,
                      max_yb]
#     print(bounding_box_b)

    width, height = max_x - min_x, max_y - min_y
    width_a, height_a = max_xa - min_xa, max_ya - min_ya
    width_b, height_b = max_xb - min_xb, max_yb - min_yb

    max_width, max_height = max(width, max_width), max(height, max_height)
    max_width_a, max_height_a = max(width_a, max_width_a), max(height_a, max_height_a)
    max_width_b, max_height_b = max(width_b, max_width_b), max(height_b, max_height_b)

    sequence_id = record['sequence_id']

    if sequence_id != previous_sequence_id:
        sequence_boundaries[previous_sequence_id] = [min_x, min_y, max_x, max_y]
        correction = get_aspect_ratio_correction(max_width, max_height)
        correction_a = get_aspect_ratio_correction(max_width_a, max_height_a)
        correction_b = get_aspect_ratio_correction(max_width_b, max_height_b)
        track_dimensions[previous_sequence_id] = [max_width, max_height, correction, max_width_a, max_height_a, correction_a, max_width_b, max_height_b, correction_b]
        # print(track_dimensions[previous_sequence_id])
        min_x = min_y = min_xa = min_ya = min_xb = min_yb = LIMIT
        max_x = max_y = max_xa = max_ya = max_xb = max_yb = -LIMIT
        max_width = max_height = max_width_a = max_height_a = max_width_b = max_height_b = -LIMIT
        previous_sequence_id = sequence_id
        
    if sequence_id not in training_sequences:
        training_sequences[sequence_id] = []
        
    steps = training_sequences[sequence_id]
    steps.append((coordinates, bounding_box, bounding_box_a, bounding_box_b))
    # print(sequence_id, bounding_box, bounding_box_a, bounding_box_b)
    
    min_x, min_y = min(min_x, bounding_box[0]), min(min_y, bounding_box[1])
    max_x, max_y = max(max_x, bounding_box[2]), max(max_y, bounding_box[3])
    
    total += 1
    
    progress_bar.update(1)
    
sequence_boundaries[previous_sequence_id] = [min_x, min_y, max_x, max_y]
correction = get_aspect_ratio_correction(max_width, max_height)
correction_a = get_aspect_ratio_correction(max_width_a, max_height_a)
correction_b = get_aspect_ratio_correction(max_width_b, max_height_b)
track_dimensions[previous_sequence_id] = [max_width, max_height, correction, max_width_a, max_height_a, correction_a, max_width_b, max_height_b, correction_b]

progress_bar.close()

print(len(training_sequences.keys()),"training sequences with", total, "steps and", len(sequence_boundaries), "boundaries")

## Hyperparameters
# BATCH_SIZE = 64
BATCH_SIZE = 256  # limit for 2080Ti (30 frames window)
# BATCH_SIZE = 512  # limit for Titan RTX (30 frames window)
LEARNING_RATE = 0.01
MINIMAL_LEARNING_RATE = 1E-9
EPOCHS = 100
WORKERS = 16
# WORKERS = 1  # Debug generator

BEST_MODEL_NAME = "crime_detection_pair_best_model.h5"

## Training
print()
print("Training model")

# take only a first few sequences to have reasonable training time on a single GPU for testing/demo purposes
generator_sequences = training_sequences
# ids = list(training_sequences.keys())[:20]  # limit for faster tests
ids = list(training_sequences.keys())
generator_sequences = {id: training_sequences[id] for id in ids}

generator = Dual_Track_Generator(generator_sequences, sequence_boundaries, track_dimensions, number_of_frames, BATCH_SIZE)

optimizer = Adam(lr=LEARNING_RATE)
base_loss_function = cumulative_point_distance_error
# base_loss_function = mean_point_distance_error

# model.compile(optimizer=optimizer, loss="mean_squared_error")
# model.compile(optimizer=optimizer, loss=pair_cumulative_point_distance_error)
# model.compile(optimizer=optimizer, loss=pair_mean_point_distance_error)
model.compile(optimizer=optimizer, loss=pair_loss)

reduce_lr = ReduceLROnPlateau(monitor="loss", factor=0.1, patience=1, verbose=1, mode="min", min_lr=MINIMAL_LEARNING_RATE)  # Hmm, doesn't work with LSTM - why?
early_stopping = EarlyStopping(monitor="loss", patience=8, verbose=1)
checkpoint = ModelCheckpoint(BEST_MODEL_NAME, monitor="loss", verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

history = model.fit_generator(generator,
                              epochs=EPOCHS,
                              workers=WORKERS,
#                               callbacks=[early_stopping, checkpoint],
                              callbacks=[reduce_lr, early_stopping, checkpoint],
                              verbose=1,
                             )
