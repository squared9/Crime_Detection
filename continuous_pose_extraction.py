import pandas as pd
import numpy as np
from collections import OrderedDict
import json
import os
import time
import sys

tracked_directory = "tracked"

clips = [
         '001', '004', '007', '008',
         '011', '012', '017',
         '020', '021', '023', '026', '028',
         '032', '037', '038', '039',
         '044', '045',
         '052', '053', '054', '057', '058',
         '061', '065', '066', '068', '069',
         '071',
         '081',
        ]

tracked_file_name = 'alphapose-results-forvis-tracked.json'

confidence_threshold = 0.0  # all keypoints with lower score should be ignored

sequences_file_name = 'sequences.csv'

# Column names for the CSV file containing all continuous pose sequences
columns = ['sequence_id',        # unique sequence ID
           'step',               # step within a sequence
           'frame',              # frame number in the original video
           'image',              # image name
           'idx',                # tracked object index
           'score',              # confidence score
           'x0', 'y0', 'c0',     # nose
           'x1', 'y1', 'c1',     # left eye
           'x2', 'y2', 'c2',     # right eye
           'x3', 'y3', 'c3',     # left ear
           'x4', 'y4', 'c4',     # right ear
           'x5', 'y5', 'c5',     # left shoulder
           'x6', 'y6', 'c6',     # right shoulder
           'x7', 'y7', 'c7',     # left elbow
           'x8', 'y8', 'c8',     # right elbow
           'x9', 'y9', 'c9',     # left wrist
           'x10', 'y10', 'c10',  # right wrist
           'x11', 'y11', 'c11',  # left hip
           'x12', 'y12', 'c12',  # right hip
           'x13', 'y13', 'c13',  # left knee
           'x14', 'y14', 'c14',  # right knee
           'x15', 'y15', 'c15',  # left ankle
           'x16', 'y16', 'c16',  # right ankle
          ]

# Types of columns of the CSV file containing all continuous pose sequences
types = {
    'sequence_id': 'int',
    'step': 'int',
    'frame': 'int',
    'image': 'object',
    'idx': 'int',
    'score': 'float',
    'x0': 'float',
    'y0': 'float',
    'c0': 'float',
    'x1': 'float',
    'y1': 'float',
    'c1': 'float',
    'x2': 'float',
    'y2': 'float',
    'c2': 'float',
    'x3': 'float',
    'y3': 'float',
    'c3': 'float',
    'x4': 'float',
    'y4': 'float',
    'c4': 'float',
    'x5': 'float',
    'y5': 'float',
    'c5': 'float',
    'x6': 'float',
    'y6': 'float',
    'c6': 'float',
    'x7': 'float',
    'y7': 'float',
    'c7': 'float',
    'x8': 'float',
    'y8': 'float',
    'c8': 'float',
    'x9': 'float',
    'y9': 'float',
    'c9': 'float',
    'x10': 'float',
    'y10': 'float',
    'c10': 'float',
    'x11': 'float',
    'y11': 'float',
    'c11': 'float',
    'x12': 'float',
    'y12': 'float',
    'c12': 'float',
    'x13': 'float',
    'y13': 'float',
    'c13': 'float',
    'x14': 'float',
    'y14': 'float',
    'c14': 'float',
    'x15': 'float',
    'y15': 'float',
    'c15': 'float',
    'x16': 'float',
    'y16': 'float',
    'c16': 'float'
}

"""
Extracts all continuous sequences from given tracking data

Arguments:
tracked_poses -- all tracking information from a single video in COCO format
sequences_df -- dataframe storing all tracked sequences
starting_id -- sequence ID for initial sequence

Returns:
sequence ID the next processing should start with
"""
def extract_sequences(tracked_poses, sequences_df, starting_id):
    last_id = starting_id
    tracked_poses = OrderedDict(sorted(tracked_poses.items(), key=lambda t: t[0]))
    tracked_idx = {}
    tracked_steps = {}
    frame = 0

    for file_name, tracks in tracked_poses.items():
        processed_tracks = set()  # keep track of discontinuities
        for track in tracks:
            keypoints = track['keypoints']
            idx = track['idx']
            score = track['scores']
            if idx not in tracked_idx:
                tracked_idx[idx] = last_id
                tracked_steps[idx] = 0
                sequence_id = last_id
                step = 0
                last_id += 1
            else:
                sequence_id = tracked_idx[idx]
                step = tracked_steps[idx] + 1
                tracked_steps[idx] = step

            processed_tracks.add(idx)

            row = [
                   sequence_id,
                   step,
                   frame,
                   file_name,
                   idx,
                   score
                  ] + keypoints

            sequences_df.loc[len(sequences_df)] = row  # Pandas is unbelievably inefficient!

        # remove all unused tracks; if a sequence has a "break", new appearance
        # of the same object should start a new sequence
        to_be_removed = []
        for idx in tracked_idx.keys():
            if idx not in processed_tracks:
                to_be_removed.append(idx)
        for idx in to_be_removed:
            tracked_idx.pop(idx, None)
            tracked_steps.pop(idx, None)

        frame += 1
    return last_id + 1

"""
Processes all files one-by-one to extract all continuous sequences
"""
def process_files():
    print("Extracting all continuous sequences")
    print("-----------------------------------")
    starting_id = 0
    sequences_df = pd.DataFrame(columns=columns)
    sequences_df = sequences_df.astype(dtype=types)
    # print(sequences_df.columns)
    for clip in clips:
        file_name = os.path.join(tracked_directory, clip, tracked_file_name)
        if os.path.isfile(file_name):
            print("Processing clip", clip)
            tracked_poses = json.load(open(file_name), object_pairs_hook=OrderedDict)
            previous_id = starting_id
            starting_id = extract_sequences(tracked_poses, sequences_df, starting_id)
            print("   ", starting_id - previous_id - 1, "sequences")
        else:
            print("Skipping clip", clip)

    # Save result
    print("Saving sequences to", sequences_file_name)
    sequences_df.sort_values(['sequence_id', 'step'], ascending=[True, True], inplace=True)
    sequences_df.to_csv(sequences_file_name, header=True, index=False)

process_files()
