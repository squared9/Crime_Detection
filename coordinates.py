import numpy as np
from enum import Enum
from data_format import bone_indices

# needed for a derivative, otherwise loss returns NaN for 0 filled tensors...
TENSOR_EPSILON = 1E-7

class CoordinateType(Enum):
    SAME = 0    # MS COCO
    OFFSET = 1  # MS COCO relative to center
    ANGLE = 2   # center + angles + length of bones

def get_coordinate_dimension(coordinate_type):
    """
    Returns (half of a) dimension of coordinates of a given type (assuming
    coordinate pairs (x, y))

    Arguments:
    coordinate_type -- type of coordinates as CoordinateType enum value
    """
    if coordinate_type == CoordinateType.SAME:
        return 17
    if coordinate_type == CoordinateType.OFFSET:
        return 18
    if coordinate_type == coordinate_type.ANGLE:
        return 13

def convert_coordinates(coordinates, coordinate_type, aspect_ratio=1.0):
    """
    Converts coordinates to either of the coordinate types

    Arguments:
    coordinates -- original coordinates; first all x, then all y values
    coordinate_type -- type of coordinates as CoordinateType enum value
    aspect_ratio -- desired adjustment of y-axis, i.e. y_new = aspect_ratio * y_old
    """
    # MS COCO
    if coordinate_type == CoordinateType.SAME:
        return coordinates

    base_number_of_coordinates = get_coordinate_dimension(CoordinateType.SAME)
    number_of_coordinates = get_coordinate_dimension(coordinate_type)
    center = np.array([np.mean(coordinates[:number_of_coordinates]), np.mean(coordinates[number_of_coordinates:])])
    # adjust y by aspect ratio
    coordinates = np.concatenate([coordinates[:base_number_of_coordinates], aspect_ratio * coordinates[base_number_of_coordinates:]]).astype(np.float32)
    # MS COCO relative to center
    if coordinate_type == CoordinateType.OFFSET:
        result = np.zeros((2 * number_of_coordinates))
        result[0] = center[0]
        result[number_of_coordinates + 1] = center[1]
        # centers = np.tile(center, number_of_coordinates)  # for (x, y) ... (x, y) input
        centers = np.repeat(center, base_number_of_coordinates)  # for (x...x, y...y) input
        # first center_x, then all xs, then center_y, then all ys
        centered_coordinates = np.subtract(coordinates, centers)
        result[1:number_of_coordinates] = centered_coordinates[:base_number_of_coordinates]
        result[number_of_coordinates + 1:] = centered_coordinates[base_number_of_coordinates:]
        return result
    # Angles and sizes of main bones + center
    if coordinate_type == coordinate_type.ANGLE:
        bone_angles = np.zeros((len(bone_indices)))
        bone_lengths = np.zeros_like(bone_angles)
        for b, indices in enumerate(bone_indices):
            if len(indices) == 2:
                i, j = indices
                p_1 = np.array([coordinates[i], coordinates[i + base_number_of_coordinates]])
                p_2 = np.array([coordinates[j], coordinates[j + base_number_of_coordinates]])
                bone = np.subtract(p_2, p_1)
            elif len(indices) == 4:
                i, j, k, l = indices
                p_a = np.array([coordinates[i], coordinates[i + base_number_of_coordinates]])
                p_b = np.array([coordinates[j], coordinates[j + base_number_of_coordinates]])
                p_c = np.array([coordinates[k], coordinates[k + base_number_of_coordinates]])
                p_d = np.array([coordinates[l], coordinates[l + base_number_of_coordinates]])
                p_1 = np.mean([p_a, p_b], axis=0)
                p_2 = np.mean([p_c, p_d], axis=0)
                bone = np.subtract(p_2, p_1)
            angle = np.arctan2(bone[1], bone[0])
            length = np.linalg.norm(bone)
            bone_angles[b] = angle
            bone_lengths[b] = length
        # first center_x, then angles, then center_y, then lengths
        result = np.concatenate([[center[0]], bone_angles, [center[1]], bone_lengths])
        return result

        # TODO: consider a loss function for angles as atan2(sin(x-y), cos(x-y))

def normalize_coordinates(coordinates, coordinate_type=CoordinateType.SAME, video_bounds=None, track_dimensions=None, prefer_track_dimension=True):
    """
    Normalizes coordinates to 0..1 in relevant sub-parts

    Arguments:
    """
    if video_bounds is None:
        return coordinates
    if video_bounds is None and track_dimensions is None:
        return coordinates
    number_of_coordinates = get_coordinate_dimension(coordinate_type)
    # track dimension has precedence over bounds during normalization if prefer_track_dimension is True
    if video_bounds is not None:
        # compute spans from video bounds
        min_x = video_bounds[0]
        min_y = video_bounds[1]
        bounds_width = video_bounds[2] - min_x
        bounds_height = video_bounds[3] - min_y
    if track_dimensions is not None:
        # compute spans from maximal track bounds
        min_x = 0
        min_y = 0
        dimension_width = track_dimensions[0]
        dimension_height = track_dimensions[1]

    if prefer_track_dimension and track_dimensions is not None or bounds is None:
        # computer factors if maximal track bounds should be used
        factors = np.zeros_like(coordinates)
        factors[:number_of_coordinates] = dimension_width
        factors[number_of_coordinates:] = dimension_height
    else:
        # computer factors if video bounds should be used
        factors[:number_of_coordinates] = bounds_width
        factors[number_of_coordinates:] = bounds_height

    shifts = np.zeros_like(coordinates)

    if video_bounds is not None:
        shifts[:number_of_coordinates] = min_x
        shifts[number_of_coordinates:] = min_y

    result = coordinates.copy()

    # perform normalization depending on coordinate type
    if coordinate_type == CoordinateType.SAME:
        # normalize plain coordinates
        result = np.subtract(result, shifts)
        result = np.divide(coordinates, factors)

    elif coordinate_type == CoordinateType.OFFSET:
        # normalize offset on [0, 1] to maximal video bounds
        if track_dimensions is not None:
            width = dimension_width
            height = dimension_height
        else:
            width = bounds_width
            height = bounds_height
        if video_bounds is not None:
            # normalize center
            result[0] = (coordinates[0] - min_x) / bounds_width
            result[number_of_coordinates] = (coordinates[number_of_coordinates] - min_y) / bounds_height
        factors = np.zeros_like(coordinates)
        factors[:number_of_coordinates] = width
        factors[number_of_coordinates:] = height
        factors[0] = factors[number_of_coordinates] = 1.0
        result = np.divide(result, factors)

    elif coordinate_type == CoordinateType.ANGLE:
        if video_bounds is not None:
            # normalize center
            result[0] = (coordinates[0] - min_x) / bounds_width
            result[number_of_coordinates] = (coordinates[number_of_coordinates] - min_y) / bounds_height
        # normalize bone lengths and keep angles as they are
        length = 1
        if track_dimensions is not None:
            length = np.linalg.norm(track_dimensions)
        elif video_bounds is not None:
            length = np.linalg.norm([bounds_width, bounds_height])

        factors = np.ones((number_of_coordinates - 1)) * length
        result[1 + number_of_coordinates:] = np.divide(coordinates[1 + number_of_coordinates:], factors)
            
    return result

def get_loss_weights(coordinate_type, ignore_movement=False, center_boost=1.0, angular_boost=1.0, bone_length_boost=1.0):
    """
    Computes weights for individual coordinate components for custom loss functions

    Arguments:
    """
    number_of_coordinates = get_coordinate_dimension(coordinate_type)
    result = np.ones((1, number_of_coordinates * 2))
    if coordinate_type == CoordinateType.OFFSET:
        if ignore_movement:
            result[0, 0] = TENSOR_EPSILON  # ignore center point prediction
            result[0, number_of_coordinates] = TENSOR_EPSILON
        else:
            result[0, 0] = center_boost
            result[0, number_of_coordinates] = center_boost
    elif coordinate_type == CoordinateType.ANGLE:
        if ignore_movement:
            result[0, 0] = TENSOR_EPSILON  # ignore center point prediction
            result[0, number_of_coordinates] = TENSOR_EPSILON
        else:
            result[0, 0] = center_boost
            result[0, number_of_coordinates] = center_boost
        # boost angular loss comparing to bone length loss
        result[0, 1: number_of_coordinates] = np.multiply(result[0, 1: number_of_coordinates], angular_boost)
        result[0, number_of_coordinates + 1:] = np.multiply(result[0, number_of_coordinates + 1:], bone_length_boost)
    return result
