# MS COCO poses structure

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

# columns in dataframe for coordinates, alternating x and y
coordinate_columns = [name + str(i) for i in range(17) for name in ['x', 'y']]

# columns in dataframe for coordinates, alternating x, y and c
coordinate_confidence_columns = [name + str(i) for i in range(17) for name in ['x', 'y', 'c']]

bone_indices = [
                ( 5,  6),  # clavicles
                ( 5,  7),  # left humerus
                ( 6,  8),  # right humerus
                ( 7,  9),  # left radius
                ( 8, 10),  # right radius
                (11, 12),  # pelvis
                (11, 13),  # left femur
                (12, 14),  # right femur
                (13, 15),  # left tibia
                (14, 16),  # right tibia
                ( 5,  6, 11, 12),  # sternum/mid-body; take average of first two and last two and connect them
                ( 5,  6,  3,  4),  # cranium/ears; take average of first two and last two and connect them
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

# Column names for the CSV file containing all continuous pose pair sequences
pair_columns = ['sequence_id',              # unique sequence ID
                'sequence_id_a',            # first pose's sequence ID
                'sequence_id_b',            # second pose's sequence ID
                'step',                     # step within a sequence
                'frame',                    # frame number in the original video
                'image',                    # image name
                'idx_a',                    # tracked object index
                'idx_b',                    # tracked object index
                'score_a',                  # confidence score
                'score_b',                  # confidence score
                'x_a0', 'y_a0', 'c_a0',     # nose, pose #1
                'x_a1', 'y_a1', 'c_a1',     # left eye, pose #1
                'x_a2', 'y_a2', 'c_a2',     # right eye, pose #1
                'x_a3', 'y_a3', 'c_a3',     # left ear, pose #1
                'x_a4', 'y_a4', 'c_a4',     # right ear, pose #1
                'x_a5', 'y_a5', 'c_a5',     # left shoulder, pose #1
                'x_a6', 'y_a6', 'c_a6',     # right shoulder, pose #1
                'x_a7', 'y_a7', 'c_a7',     # left elbow, pose #1
                'x_a8', 'y_a8', 'c_a8',     # right elbow, pose #1
                'x_a9', 'y_a9', 'c_a9',     # left wrist, pose #1
                'x_a10', 'y_a10', 'c_a10',  # right wrist, pose #1
                'x_a11', 'y_a11', 'c_a11',  # left hip, pose #1
                'x_a12', 'y_a12', 'c_a12',  # right hip, pose #1
                'x_a13', 'y_a13', 'c_a13',  # left knee, pose #1
                'x_a14', 'y_a14', 'c_a14',  # right knee, pose #1
                'x_a15', 'y_a15', 'c_a15',  # left ankle, pose #1
                'x_a16', 'y_a16', 'c_a16',  # right ankle, pose #1
                'x_b0', 'y_b0', 'c_b0',     # nose, pose #2
                'x_b1', 'y_b1', 'c_b1',     # left eye, pose #2
                'x_b2', 'y_b2', 'c_b2',     # right eye, pose #2
                'x_b3', 'y_b3', 'c_b3',     # left ear, pose #2
                'x_b4', 'y_b4', 'c_b4',     # right ear, pose #2
                'x_b5', 'y_b5', 'c_b5',     # left shoulder, pose #2
                'x_b6', 'y_b6', 'c_b6',     # right shoulder, pose #2
                'x_b7', 'y_b7', 'c_b7',     # left elbow, pose #2
                'x_b8', 'y_b8', 'c_b8',     # right elbow, pose #2
                'x_b9', 'y_b9', 'c_b9',     # left wrist, pose #2
                'x_b10', 'y_b10', 'c_b10',  # right wrist, pose #2
                'x_b11', 'y_b11', 'c_b11',  # left hip, pose #2
                'x_b12', 'y_b12', 'c_b12',  # right hip, pose #2
                'x_b13', 'y_b13', 'c_b13',  # left knee, pose #2
                'x_b14', 'y_b14', 'c_b14',  # right knee, pose #2
                'x_b15', 'y_b15', 'c_b15',  # left ankle, pose #2
                'x_b16', 'y_b16', 'c_b16',  # right ankle, pose #2
                ]

# columns in dataframe for pair coordinates, alternating x and y for first pose, followed by second pose
pair_coordinate_columns = [name + str(i) for i in range(17) for name in ['x_a', 'y_a']] + [name + str(i) for i in range(17) for name in ['x_b', 'y_b']]

# columns in dataframe for pair coordinates with confidence, alternating x, y and c for first pose, followed by second pose
pair_coordinate_confidence_columns = [name + str(i) for i in range(17) for name in ['x_a', 'y_a', 'c_a']] + [name + str(i) for i in range(17) for name in ['x_b', 'y_b', 'c_b']]

# Types of columns of the CSV file containing all continuous pair pose sequences
pair_types = {
    'sequence_id': 'int',
    'sequence_id_a': 'int',
    'sequence_id_b': 'int',
    'step': 'int',
    'frame': 'int',
    'image': 'object',
    'idx_a': 'int',
    'idx_b': 'int',
    'score_a': 'float',
    'score_b': 'float',
    'x_a0': 'float',
    'y_a0': 'float',
    'c_a0': 'float',
    'x_a1': 'float',
    'y_a1': 'float',
    'c_a1': 'float',
    'x_a2': 'float',
    'y_a2': 'float',
    'c_a2': 'float',
    'x_a3': 'float',
    'y_a3': 'float',
    'c_a3': 'float',
    'x_a4': 'float',
    'y_a4': 'float',
    'c_a4': 'float',
    'x_a5': 'float',
    'y_a5': 'float',
    'c_a5': 'float',
    'x_a6': 'float',
    'y_a6': 'float',
    'c_a6': 'float',
    'x_a7': 'float',
    'y_a7': 'float',
    'c_a7': 'float',
    'x_a8': 'float',
    'y_a8': 'float',
    'c_a8': 'float',
    'x_a9': 'float',
    'y_a9': 'float',
    'c_a9': 'float',
    'x_a10': 'float',
    'y_a10': 'float',
    'c_a10': 'float',
    'x_a11': 'float',
    'y_a11': 'float',
    'c_a11': 'float',
    'x_a12': 'float',
    'y_a12': 'float',
    'c_a12': 'float',
    'x_a13': 'float',
    'y_a13': 'float',
    'c_a13': 'float',
    'x_a14': 'float',
    'y_a14': 'float',
    'c_a14': 'float',
    'x_a15': 'float',
    'y_a15': 'float',
    'c_a15': 'float',
    'x_a16': 'float',
    'y_a16': 'float',
    'c_a16': 'float',
    'x_a0': 'float',
    'y_a0': 'float',
    'c_a0': 'float',
    'x_a1': 'float',
    'y_a1': 'float',
    'c_a1': 'float',
    'x_a2': 'float',
    'y_a2': 'float',
    'c_a2': 'float',
    'x_a3': 'float',
    'y_a3': 'float',
    'c_a3': 'float',
    'x_a4': 'float',
    'y_a4': 'float',
    'c_a4': 'float',
    'x_a5': 'float',
    'y_a5': 'float',
    'c_a5': 'float',
    'x_a6': 'float',
    'y_a6': 'float',
    'c_a6': 'float',
    'x_a7': 'float',
    'y_a7': 'float',
    'c_a7': 'float',
    'x_a8': 'float',
    'y_a8': 'float',
    'c_a8': 'float',
    'x_a9': 'float',
    'y_a9': 'float',
    'c_a9': 'float',
    'x_a10': 'float',
    'y_a10': 'float',
    'c_a10': 'float',
    'x_a11': 'float',
    'y_a11': 'float',
    'c_a11': 'float',
    'x_a12': 'float',
    'y_a12': 'float',
    'c_a12': 'float',
    'x_a13': 'float',
    'y_a13': 'float',
    'c_a13': 'float',
    'x_a14': 'float',
    'y_a14': 'float',
    'c_a14': 'float',
    'x_a15': 'float',
    'y_a15': 'float',
    'c_a15': 'float',
    'x_a16': 'float',
    'y_a16': 'float',
    'c_a16': 'float'
}
