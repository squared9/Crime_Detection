import pandas as pd
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
import json
import os
import time
import sys

from data_format import columns, types, coordinate_columns, coordinate_confidence_columns, pair_columns, pair_types, pair_coordinate_columns, pair_coordinate_confidence_columns

# USE_DATA_COLUMNS = False  # use slow DataFrame
USE_DATA_COLUMNS = True   # use column dictionary

confidence_threshold = 0.0  # all keypoints with lower score should be ignored

sequences_file_name = 'sequences.csv'
pose_pair_sequences_file_name = "pose_pair_sequences.csv"

"""
Retrieves records from dataframe related to a given image prefix and frame
REMARK: Not used

Arguments:
sequences -- dataframe with sequences
image_prefix -- prefix of image files, e.g. 001-23094.png has a prefix of 001
frame -- frame number to be extracted
"""
def get_records_from_frame(sequences, image_prefix, frame):
    records = sequences[sequences['image'].startswith(image_prefix) & sequences['frame'] == frame]
    return records

"""
Adds pose pair records to a dataframe

Arguments:
data_df -- dataframe for pose pair sequences
records -- entry in the form of (sequence_id, id_1, id_2, start_frame, step, end_frame, record_1, record_2)
           as defined in the sequence spatio-temporal search
"""
def add_records(data_df, records):
    for record in records:
        id, id_1, id_2, start_frame, step, _, record_1, record_2 = record
        keypoints_1 = record_1[coordinate_confidence_columns].values
        keypoints_2 = record_2[coordinate_confidence_columns].values
        keypoints_1, keypoints_2 = list(keypoints_1), list(keypoints_2)
        row = [
                id,
                id_1,
                id_2,
                step,
                start_frame + step,
                record_1['image'],
                record_1['idx'],
                record_2['idx'],
                record_1['score'],
                record_2['score'],
                ] + keypoints_1 + keypoints_2

        # Pandas is catastrophically badly written for dynamic data/append
        # Just not appending rows makes this script finish in 4 minutes
        # When appending rows is enabled, it's 4+ hours
        data_df.loc[len(data_df)] = row

"""
Appends pose pair records to a column-based dictionary
REMARK: 26x faster than adding records directly to dataframe (thanks Pandas! :-/ )

Arguments:
columns -- dataframe for pose pair sequences
records -- entry in the form of (sequence_id, id_1, id_2, start_frame, step, end_frame, record_1, record_2)
           as defined in the sequence spatio-temporal search
"""
def append_records(data_columns, records):
    for record in records:
        id, id_1, id_2, start_frame, step, _, record_1, record_2 = record
        keypoints_1 = record_1[coordinate_confidence_columns].values
        keypoints_2 = record_2[coordinate_confidence_columns].values
        keypoints_1, keypoints_2 = list(keypoints_1), list(keypoints_2)
        row = [
                id,
                id_1,
                id_2,
                step,
                start_frame + step,
                record_1['image'],
                record_1['idx'],
                record_2['idx'],
                record_1['score'],
                record_2['score'],
                ] + keypoints_1 + keypoints_2

        for i, column in enumerate(pair_columns):
            if column not in data_columns:
                data_columns[column] = []
            values = data_columns[column]
            values.append(row[i])

"""
Extracts all continuous pose pair sequences from given tracking data

Arguments:
sequences_df -- dataframe storing all tracked sequences
prefixes -- all image prefixes; one prefix per video, processed one by one
maximal_radius -- maximal distance between centers of two poses to be considered
                  spatially related; any tracking is interrupted when distance 
                  exceeds this value
minimal_duration -- the lowest number of frames to consider two nearby tracked poses
                    temporally related

Returns:
Dataframe with tracked pose pairs
"""
def extract_pose_pair_sequences(sequences_df, prefixes, maximal_radius, minimal_duration, verbose=False):
    if USE_DATA_COLUMNS:
        data_columns = {}
    else:
        result_df = pd.DataFrame(columns=pair_columns)
        result_df = result_df.astype(dtype=pair_types)

    current_ongoing_sequences = {}  # keep track of currently ongoing sequences
    all_recorded_sequences = {}  # keep track of currently ongoing sequences that lasted minimal_duration already

    progress_bar = tqdm(total=len(prefixes))

    last_persistent_sequence_id = 0

    for i, prefix in enumerate(prefixes):
        if verbose:
            print("processing prefix", prefix)
        
        records_df = sequences_df[sequences_df['image'].str.startswith(prefix)]
        if verbose:
            print(len(records_df), "to process")

        min_frame, max_frame = records_df['frame'].min(), records_df['frame'].max()

        for frame in range(min_frame, max_frame + 1):
            parallel_sequences_df = records_df[records_df['frame'] == frame]
            parallel_sequences = {}
            for j, record in parallel_sequences_df.iterrows():
                sequence_id = record["sequence_id"]
                coordinates = record[coordinate_columns].values.astype(np.float32)
                center = get_center(coordinates)
                parallel_sequences[sequence_id] = (center, coordinates, record)

            ids = list(parallel_sequences.keys())
            for id_1 in ids:
                center_1 = parallel_sequences[id_1][0]
                record_1 = parallel_sequences[id_1][2]
                for id_2 in ids:
                    if id_1 >= id_2:
                        continue
                    center_2 = parallel_sequences[id_2][0]
                    record_2 = parallel_sequences[id_2][2]
                    tid_1 = min(id_1, id_2)
                    tid_2 = max(id_1, id_2)
                    tracked_id = (tid_1, tid_2)
                    # are tracked centers close enough to be interesting?
                    if center_distance(center_1, center_2) <= maximal_radius:
                        if tracked_id not in current_ongoing_sequences:
                            # if it is a new sequence to track
                            current_ongoing_sequences[tracked_id] = [[tid_1, tid_2, frame, 0, -1, record_1, record_2]]
                        elif tracked_id not in all_recorded_sequences:
                            # is the tracked sequence long enough to become permanent?
                            start_frame = current_ongoing_sequences[tracked_id][-1][2]
                            current_ongoing_sequences[tracked_id].append([tid_1, tid_2, start_frame, frame - start_frame, -1, record_1, record_2])
                            duration = frame - start_frame + 1
                            if duration >= minimal_duration:
                                tracked_records = current_ongoing_sequences[tracked_id]
                                # prepend new sequence id
                                tracked_records = [tuple([last_persistent_sequence_id] + entry) for entry in tracked_records]
                                all_recorded_sequences[tracked_id] = tracked_records
                                last_persistent_sequence_id += 1
                        else:
                            # add new entry to tracked sequences
                            id, _, _, start_frame, current_frame, _, _, _ = all_recorded_sequences[tracked_id][-1]
                            entry = (id, tid_1, tid_2, start_frame, current_frame + 1, -1, record_1, record_2)
                            all_recorded_sequences[tracked_id].append(entry)
                    else:
                        # remove from currently tracked sequences
                        if tracked_id in current_ongoing_sequences:
                            current_ongoing_sequences.pop(tracked_id)
                        # flush persistent sequences to result
                        if tracked_id in all_recorded_sequences:
                            # TODO: save to result_df
                            recorded_sequence = all_recorded_sequences.pop(tracked_id)
                            id, tid_1, tid_2, start_frame, current_frame, _, _, _ = recorded_sequence[-1]
                            if USE_DATA_COLUMNS:
                                append_records(data_columns, recorded_sequence)
                            else:
                                add_records(result_df, recorded_sequence)
                            if verbose:
                                print("Storing sequence", id, tracked_id, "duration:", frame - start_frame + 1)

        # flush all persistent sequences
        tracked_ids = list(all_recorded_sequences.keys())
        for tracked_id in tracked_ids:
            if tracked_id in all_recorded_sequences:
                # TODO: save to result_df
                recorded_sequence = all_recorded_sequences.pop(tracked_id)
                id, tid_1, tid_2, start_frame, current_frame, _, _, _ = recorded_sequence[-1]
                if USE_DATA_COLUMNS:
                    append_records(data_columns, recorded_sequence)
                else:
                    add_records(result_df, recorded_sequence)
                if verbose:
                    print("Storing sequence", id, tracked_id, "duration:", frame - start_frame + 1)

        progress_bar.update(1)

    progress_bar.close()

    print(last_persistent_sequence_id, "pose pair sequences found")

    if USE_DATA_COLUMNS:
        return data_columns
    else:
        return result_df

"""
Arguments:
coordinates -- array of x, y pairs; x at even and y at odd positions

Returns:
Center/mean of coordinates
"""
def get_center(coordinates):
    return np.array(np.mean(coordinates[::2]), np.mean(coordinates[1::2]))

"""
Computes Euclidean distance of two pose centers

Arguments:
center_1 -- first center as (x1, y1)
center_2 -- second center as (x2, y2)
"""
def center_distance(center_1, center_2):
    return np.linalg.norm(center_1 - center_2)

"""
Computes Euclidean distance of centers of two sets of pose coordinates

Arguments:
coordinates_1 -- first set of 17 MS COCO coordinates
coordinates_2 -- second set of 17 MS COCO coordinates
"""
def coordinate_distance(coordinates_1, coordinates_2):
    center_1 = get_center(coordinates_1)
    center_2 = get_center(coordinates_2)
    return center_distance(center_1, center_2)

"""
Returns all image prefixes of images in tracking dataframe

Arguments:
sequences_df -- dataframe containing individual tracked poses
"""
def get_image_prefixes(sequences_df):
    image_file_names = sequences_df['image'].values
    image_file_names = [x[:3] for x in image_file_names]
    prefixes = sorted(list(set(image_file_names)))
    return prefixes

"""
Processes all files one-by-one to extract all continuous sequences
"""
def process_pose_pair_sequences(maximal_radius=50, minimal_duration=24):
    print("Extracting all sequences of related pairs of poses")
    print("--------------------------------------------------")

    sequences_df = pd.read_csv(sequences_file_name, sep=",", header=0, index_col=None)
    sequences_df = sequences_df.astype(dtype=types)
    # sort by video/frame/sequence_id
    sequences_df.sort_values(['image', 'frame', 'sequence_id'], ascending=[True, True, True], inplace=True)

    prefixes = get_image_prefixes(sequences_df)

    pose_pair_sequences_df = extract_pose_pair_sequences(sequences_df, prefixes, maximal_radius, minimal_duration, False)

    # Save result
    print("Saving sequences to", pose_pair_sequences_file_name)
    if USE_DATA_COLUMNS:
        pose_pair_sequences_df = pd.DataFrame.from_dict(pose_pair_sequences_df)
        pose_pair_sequences_df.reindex(columns=pair_columns)

    pose_pair_sequences_df.sort_values(['sequence_id', 'step'], ascending=[True, True], inplace=True)
    pose_pair_sequences_df.to_csv(pose_pair_sequences_file_name, header=True, index=False)

MAXIMAL_RADIUS = 50  # maximal center distance
MINIMAL_DURATION = 24  # minimal temporal interaction duration

process_pose_pair_sequences(MAXIMAL_RADIUS, MINIMAL_DURATION)
