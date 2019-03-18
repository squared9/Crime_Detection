"""
Loss function definitions
"""
import numpy as np
import keras.backend as K
import tensorflow as tf
from coordinates import get_loss_weights
from coordinates import CoordinateType
from coordinates import get_coordinate_dimension
# NOTE: cordinate_encoding variable must be set to proper coordinate type
#       Keras can't pass additional parameters to its loss function callbacks :-(
from configuration import IGNORE_MOVEMENT, coordinate_encoding, number_of_coordinates, number_of_frames

loss_weight_adjustments = get_loss_weights(coordinate_encoding, 
                                           IGNORE_MOVEMENT, 
                                           center_boost=1/100.0,
                                           angular_boost=1.0, 
                                           bone_length_boost=1.0)

loss_weight_adjustments = np.repeat(loss_weight_adjustments, number_of_frames, axis=0)

loss_weight_adjustments = K.variable(value=loss_weight_adjustments)

def cumulative_point_distance_error(y_true, y_pred):
    """
    Cumulative Euclidean distance of all (x, y) pairs between true and predicted vectors
    Depending on coordinate encoding, it could be either distances of (x, y) pairs or
    centers with distances of relative (dx, dy) pairs, or centers with angular
    and bone length differences
    
    Arguments:
    y_true -- tensor of true values, [batch number, frame number, x/y as 0/1, coordinate]
    y_pred -- tensor of predicted values, [batch number, frame number, x/y as 0/1, coordinate]
    """
    
    # NOTE: Loss functions are defined in TensorFlow as Keras' backend doesn't expose many
    #       of the necessary functions :-(

    if coordinate_encoding == CoordinateType.SAME:
        delta_x = y_true[:, :, 0, :] - y_pred[:, :, 0, :]
        delta_y = y_true[:, :, 1, :] - y_pred[:, :, 1, :]

        weight_x = loss_weight_adjustments[:, :, :number_of_coordinates]
        weight_y = loss_weight_adjustments[:, :, number_of_coordinates:]
       
        # loss is sum of square roots of sums of squares per rows of weight-adjusted x and y differences
        loss = tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(tf.multiply(delta_x, weight_x)), axis=2))) +\
               tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(tf.multiply(delta_y, weight_y)), axis=2)))
        
    elif coordinate_encoding == CoordinateType.OFFSET:
        # (c_x, c_y, x_0...x_n, y_0...y_n)
        delta_x = y_true[:, :, 0, 1:] - y_pred[:, :, 0, 1:]
        delta_y = y_true[:, :, 1, 1:] - y_pred[:, :, 1, 1:]
        c_delta_x = y_true[:, :, 0, 0] - y_pred[:, :, 0, 0]
        c_delta_y = y_true[:, :, 1, 0] - y_pred[:, :, 1, 0]

        weight_x = loss_weight_adjustments[1:number_of_coordinates]
        weight_y = loss_weight_adjustments[number_of_coordinates + 1:]
        weight_c_x = loss_weight_adjustments[:, 0]
        weight_c_y = loss_weight_adjustments[:, 1]

        # relative coordinate loss + center loss
        loss = tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(tf.multiply(delta_x, weight_x)), axis=2))) +\
               tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(tf.multiply(delta_y, weight_y)), axis=2))) +\
               tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(tf.multiply(c_delta_x, weight_c_x)), axis=1))) +\
               tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(tf.multiply(c_delta_y, weight_c_y)), axis=1)))

    elif coordinate_encoding == CoordinateType.ANGLE:
        # WARNING: This loss function is extremely slow !!!
        # Use only on Multi-GPU configurations (>= 8x) for a reasonable convergence time
        
        # input is: [batch x frame x dimension x coordinate]

        # angular distances, normalized to [-pi, pi]
        angle_true = y_true[:, :, 0, 1:]
        angle_pred = y_pred[:, :, 0, 1:]
        delta_angle = tf.atan2(tf.sin(angle_true - angle_pred), tf.cos(angle_true - angle_pred))

        # bone length differences
        delta_bone_length = y_true[:, :, 1, 1:] - y_pred[:, :, 1, 1:]
        
        # center position differences
        c_delta_x = y_true[:, :, 0, 0] - y_pred[:, :, 0, 0]
        c_delta_y = y_true[:, :, 1, 0] - y_pred[:, :, 1, 0]

        # weights for angles
        weight_angle = loss_weight_adjustments[:, 1: number_of_coordinates]
        
        # weights for bone lengths
        weight_bone_length = loss_weight_adjustments[:, number_of_coordinates + 1:]
        
        # center weight
        weight_c_x = loss_weight_adjustments[:, 0]
        weight_c_y = loss_weight_adjustments[:, 1]
        
        # loss is sum of squares of weight-adjusted angular differences + sum of squares of weight-adjusted
        # bone length differences + sum of squares of weight-adjusted center differences;
        # this is done per row (row = batch x frame), then summed together
        loss = tf.reduce_sum(
                             tf.reduce_sum(tf.square(tf.multiply(delta_angle, weight_angle)), axis=2) +\
                             tf.reduce_sum(tf.square(tf.multiply(delta_bone_length, weight_bone_length)), axis=2)
                            ) +\
               tf.reduce_sum(tf.sqrt(
                                     tf.reduce_sum(tf.square(tf.multiply(c_delta_x, weight_c_x)), axis=1) +\
                                     tf.reduce_sum(tf.square(tf.multiply(c_delta_y, weight_c_y)), axis=1)
                                    )
                            )
    return loss

def mean_point_distance_error(y_true, y_pred):
    """
    Mean Euclidean distance of all (x, y) pairs between true and predicted vectors
    Depending on coordinate encoding, it could be either distances of (x, y) pairs or
    centers with distances of relative (dx, dy) pairs, or centers with angular
    and bone length differences
    
    Arguments:
    y_true -- tensor of true values, [batch number, frame number, x/y as 0/1, coordinate]
    y_pred -- tensor of predicted values, [batch number, frame number, x/y as 0/1, coordinate]
    """
    if coordinate_encoding == CoordinateType.SAME:
        delta_x = y_true[:, :, 0, :] - y_pred[:, :, 0, :]
        delta_y = y_true[:, :, 1, :] - y_pred[:, :, 1, :]

        weight_x = loss_weight_adjustments[:, :, :number_of_coordinates]
        weight_y = loss_weight_adjustments[:, :, number_of_coordinates:]
       
        # loss is sum of square roots of sums of squares per rows of weight-adjusted x and y differences
        loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.multiply(delta_x, weight_x)), axis=2))) +\
               tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.multiply(delta_y, weight_y)), axis=2)))
        
    elif coordinate_encoding == CoordinateType.OFFSET:
        # (c_x, c_y, x_0...x_n, y_0...y_n)
        delta_x = y_true[:, :, 0, 1:] - y_pred[:, :, 0, 1:]
        delta_y = y_true[:, :, 1, 1:] - y_pred[:, :, 1, 1:]
        c_delta_x = y_true[:, :, 0, 0] - y_pred[:, :, 0, 0]
        c_delta_y = y_true[:, :, 1, 0] - y_pred[:, :, 1, 0]

        weight_x = loss_weight_adjustments[1: number_of_coordinates]
        weight_y = loss_weight_adjustments[number_of_coordinates + 1:]
        weight_c_x = loss_weight_adjustments[:, 0]
        weight_c_y = loss_weight_adjustments[:, 1]

        # relative coordinate loss + center loss
        loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.multiply(delta_x, weight_x)), axis=2))) +\
               tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.multiply(delta_y, weight_y)), axis=2))) +\
               tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.multiply(c_delta_x, weight_c_x)), axis=1))) +\
               tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.multiply(c_delta_y, weight_c_y)), axis=1)))

    elif coordinate_encoding == CoordinateType.ANGLE:
        # WARNING: This loss function is extremely slow !!!
        # Use only on Multi-GPU configurations (>= 8x) for a reasonable convergence time
        
        # input is: [batch x frame x dimension x coordinate]

        # angular distances, normalized to [-pi, pi]
        angle_true = y_true[:, :, 0, 1:]
        angle_pred = y_pred[:, :, 0, 1:]
        delta_angle = tf.atan2(tf.sin(angle_true - angle_pred), tf.cos(angle_true - angle_pred))

        # bone length differences
        delta_bone_length = y_true[:, :, 1, 1:] - y_pred[:, :, 1, 1:]
        
        # center position differences
        c_delta_x = y_true[:, :, 0, 0] - y_pred[:, :, 0, 0]
        c_delta_y = y_true[:, :, 1, 0] - y_pred[:, :, 1, 0]

        # weights for angles
        weight_angle = loss_weight_adjustments[:, 1: number_of_coordinates]
        
        # weights for bone lengths
        weight_bone_length = loss_weight_adjustments[:, number_of_coordinates + 1:]
        
        # center weight
        weight_c_x = loss_weight_adjustments[:, 0]
        weight_c_y = loss_weight_adjustments[:, 1]
        
        # loss is sum of squares of weight-adjusted angular differences + sum of squares of weight-adjusted
        # bone length differences + sum of squares of weight-adjusted center differences;
        # this is done per row (row = batch x frame), then summed together
        loss = tf.reduce_mean(
                             tf.reduce_sum(tf.square(tf.multiply(delta_angle, weight_angle)), axis=2) +\
                             tf.reduce_sum(tf.square(tf.multiply(delta_bone_length, weight_bone_length)), axis=2)
                            ) * 0.5 +\
               tf.reduce_mean(tf.sqrt(
                                     tf.reduce_sum(tf.square(tf.multiply(c_delta_x, weight_c_x)), axis=1) +\
                                     tf.reduce_sum(tf.square(tf.multiply(c_delta_y, weight_c_y)), axis=1)
                                    )
                            ) * 0.5
    return loss

# NOTE: This needs to be set for a pair-wise loss function
#       Keras can't pass additional parameters to its loss function callbacks :-(
loss_function = cumulative_point_distance_error

def get_pair_loss(y_true, y_pred):
    """
    Computes a pair-wise loss function assuming two parameters 
    Loss function for a single track is defined by loss_function variable
    It assumes coordinates as [batch x frame x dimension x 2 * number of coordinates],
    where the last two dimensions have concatenated coordinates of both tracked poses
    The resulting loss is just a simple sum of two individual losses on their corresponding
    coordinate portion

    Arguments:
    y_true -- tensor of true values, [batch number, frame number, x/y as 0/1, coordinate]
    y_pred -- tensor of predicted values, [batch number, frame number, x/y as 0/1, coordinate]

    """
    loss_1 = loss_function(y_true[:, :, :, :number_of_coordinates],
                           y_pred[:, :, :, :number_of_coordinates])
    loss_2 = loss_function(y_true[:, :, :, number_of_coordinates:],
                           y_pred[:, :, :, number_of_coordinates:])
    return tf.reduce_sum(loss_1, loss_2)
