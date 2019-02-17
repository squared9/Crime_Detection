import pandas as pd
import numpy as np
from collections import OrderedDict
import json
import os
import time
import sys

from data_format import columns, types

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
