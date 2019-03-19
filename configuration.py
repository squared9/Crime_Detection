from coordinates import CoordinateType, get_coordinate_dimension

NORMALIZE_COORDINATES = True    # normalize them to [0; 1] based on largest bounding box of a tracked object
# NORMALIZE_COORDINATES = False   # keep coordinates as they are

NORMALIZE_ASPECT_RATIO = True    # normalize coordinates to expected standing human body ratio
# NORMALIZE_ASPECT_RATIO = False   # keep human body ratio as is

IDEAL_HUMAN_BODY_RATIO = 3.5/1   # average human height / width

IGNORE_MOVEMENT = True   # ignore center in loss function
# IGNORE_MOVEMENT = False  # keep center in loss function

# should autoencoder be first in the model instead of RNN/attention transformer?
AUTOENCODER_FIRST = True
# AUTOENCODER_FIRST = False

# TODO: implement attention transformer instead of RNN
USE_ATTENTION = False
# USE_ATTENTION = True

# coordinate encoding; choose which type to use
# coordinate_encoding = CoordinateType.SAME  # just keep coordinates
# coordinate_encoding = CoordinateType.OFFSET  # center point + offsets to it
coordinate_encoding = CoordinateType.ANGLE  # encode bone angles and lengths

# number of coordinates
number_of_coordinates = get_coordinate_dimension(coordinate_encoding)
base_number_of_coordinates = get_coordinate_dimension(CoordinateType.SAME)

# length of a training sequence in frames
# number_of_frames = 300
number_of_frames = 30
