import numpy as np

from coordinates import CoordinateType, get_coordinate_dimension

MAXIMAL_SCALE = 3.0

def augment_coordinates(coordinates, 
                        randomize_positions=True, 
                        forced_shifts=None, 
                        p_horizontal_flip=0.5, 
                        force_flip=None,
                        mean_scale_factor=1.0,
                        force_scale=None,
                        sequence_boundary=None, 
                        number_of_coordinates=get_coordinate_dimension(CoordinateType.SAME)
                       ):
    """
    """

    result = coordinates.copy()
    # horizontal flip
    if force_flip is not None:
        do_flip = force_flip
    elif p_horizontal_flip > 0:
        do_flip = np.random.random()
        if do_flip > p_horizontal_flip:
            do_flip = True
    if do_flip:
        if sequence_boundary is not None:
            min_x, min_y, max_x, max_y = sequence_boundary
            result[:number_of_coordinates] = max_x + min_x - result[:number_of_coordinates]

    # shift coordinates by random amount within sequence frame applied to upper-left corner
    shift_x = shift_y = 0
    if forced_shifts is not None:
        shift_x, shift_y = forced_shifts
    elif randomize_positions:
        if sequence_boundary is not None:
            # shift within sequence boundary if provided
            min_x, min_y, max_x, max_y = sequence_boundary
            shift_x = np.random.random() * (max_x - min_x) + min_x
            shift_y = np.random.random() * (max_y - min_y) + min_y
        shifts_y = [shift_x] * number_of_coordinates 
        shifts_x = [shift_y] * number_of_coordinates
        shifts = np.concatenate([np.array(shifts_x), np.array(shifts_y)])
        result = np.add(result, shifts)

    # rescale by random amount (at most MAXIMAL_SCALEx + mean_scale_factor)
    scale = mean_scale_factor
    if force_scale is not None:
        scale = force_scale
    elif mean_scale_factor != 1.0:
        scale = np.random.random() * MAXIMAL_SCALE + mean_scale_factor
    result = np.multiply(result, scale)

    return result

def augment_pair_coordinates(coordinates_1,
                             coordinates_2,
                             randomize_positions=True, 
                             forced_shifts=None, 
                             p_horizontal_flip=0.5, 
                             force_flip=None,
                             mean_scale_factor=1.0,
                             force_scale=None,
                             sequence_boundary=None, 
                             number_of_coordinates=get_coordinate_dimension(CoordinateType.SAME)
                            ):
    """
    """

    # coordinates_1 = coordinates[: 2 * number_of_coordinates]
    # coordinates_2 = coordinates[2 * number_of_coordinates:]

    # force_* variables are used to make sure both coordinates in a pair are augmented the same way

    # flip horizontally?
    if force_flip is None and p_horizontal_flip > 0:
        force_flip = np.random.random()
        force_flip = force_flip > p_horizontal_flip

    # shift by how much?
    if forced_shifts is None and sequence_boundary is not None:
        min_x, min_y, max_x, max_y = sequence_boundary
        shift_x = np.random.random() * (max_x - min_x) + min_x
        shift_y = np.random.random() * (max_y - min_y) + min_y
        forced_shifts = (shift_x, shift_y)

    # scale by how much?
    if force_scale is not None and mean_rescale_factor != 1.0:
        force_scale = np.random.random() * MAXIMAL_SCALE + mean_scale_factor

    coordinates_1 = augment_coordinates(coordinates_1, force_flip=force_flip, forced_shifts=forced_shifts, force_scale=force_scale)
    coordinates_2 = augment_coordinates(coordinates_2, force_flip=force_flip, forced_shifts=forced_shifts, force_scale=force_scale)

    # result = np.concatenate([coordinates_1, coordinates_2])
    result = coordinates_1, coordinates_2

    return result
