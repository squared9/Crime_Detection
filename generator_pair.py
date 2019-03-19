import numpy as np

from keras.utils import Sequence
from coordinates import normalize_coordinates
from augmentation import augment_coordinates

from configuration import coordinate_encoding, number_of_coordinates, base_number_of_coordinates, number_of_frames, NORMALIZE_ASPECT_RATIO, NORMALIZE_COORDINATES
from coordinates import convert_coordinates, normalize_coordinates
from augmentation import augment_pair_coordinates 

class Dual_Track_Generator(Sequence):
    
    def __init__(self, sequences, sequence_boundaries, track_dimensions, number_of_frames, batch_size=1, 
                 randomized=True, randomize_positions=False, mean_scale_factor=1.0, p_horizontal_flip=0.5):
        """
        Initializes generator
        
        Arguments:
        sequences -- training sequences dictionary, sequence_id -> [(coordinates, bounding box) for each step]
        sequence_boundaries -- dictionary with bounding boxes of whole sequences, sequence_id -> bounding box
        number_of_frames -- number of frames for a single training sample, e.g. 300 frames
        batch_size -- training batch size
        randomized -- if true, order of sequences is randomized in each epoch
        """
        self.sequences = sequences
        self.sequence_boundaries = sequence_boundaries
        self.track_dimensions = track_dimensions
        self.number_of_frames = number_of_frames
        # TODO: allow/test batch_size > 1
        self.batch_size = batch_size
        self.ids = list(sequences.keys())
#         self.dimension = len(sequences[self.ids[0]][0][0]) // 2  # length of coordinates
        self.dimension = number_of_coordinates
        self.randomized = randomized
        if self.randomized:
            self.ids = np.random.permutation(self.ids)
        self.randomize_positions = randomize_positions
        self.mean_scale_factor = mean_scale_factor
        self.p_horizontal_flip = p_horizontal_flip
        self.index = 0
        self.length = None
        self.steps = {}  # index to sequence_id + offset for each training step

    def __len__(self):
        """
        Returns number of batches during a single full pass
        """
        offset = 0
        if self.length is None:
            self.length = 0
            for id in self.ids:
                number_of_subsequences = len(self.sequences[id]) - self.number_of_frames + 1
                self.length += number_of_subsequences
                # for each possible subsequence of a sequence add a new (id, starting step) record
                # to allow random batch retrieval with arbitrary size
                for i in range(number_of_subsequences):
                    self.steps[offset] = (id, i)  # (sequence_id, offset)
                    offset += 1
            self.length = np.floor(self.length / float(self.batch_size))  # skip last incomplete batch
            self.length = int(self.length)
                
        return self.length

    def __getitem__(self, idx):
        """
        Receives a new training batch with index idx
        """
        x = np.ndarray((self.batch_size, self.number_of_frames, 2, 2 * self.dimension), dtype=np.float32)
        index = idx * self.batch_size
        for j in range(self.batch_size):
            for i in range(self.number_of_frames):
                sequence_id, step = self.steps[index]
                coordinates_a = self.sequences[sequence_id][step][0][:2 * base_number_of_coordinates]
                coordinates_b = self.sequences[sequence_id][step][0][2 * base_number_of_coordinates:]
                aspect_ratio = 1.0
                if NORMALIZE_ASPECT_RATIO:
                    # take average of both aspect ratios for now
                    # print(len(self.track_dimensions[sequence_id]))
                    aspect_ratio = np.mean([self.track_dimensions[sequence_id][5], self.track_dimensions[sequence_id][8]])
                coordinates_a, coordinates_b = self.perform_augmentation(sequence_id, coordinates_a, coordinates_b)
                coordinates_a = convert_coordinates(coordinates_a, coordinate_encoding, aspect_ratio)
                coordinates_b = convert_coordinates(coordinates_b, coordinate_encoding, aspect_ratio)
                if NORMALIZE_COORDINATES:
                    coordinates_a = normalize_coordinates(coordinates_a, 
                                                          coordinate_encoding, 
                                                          self.sequence_boundaries[sequence_id],
                                                          self.track_dimensions[sequence_id][:3],
                                                          prefer_track_dimension=True,
                                                         )
                    coordinates_b = normalize_coordinates(coordinates_b, 
                                                          coordinate_encoding, 
                                                          self.sequence_boundaries[sequence_id],
                                                          self.track_dimensions[sequence_id][:3],
                                                          prefer_track_dimension=True,
                                                         )
                x[j, i, :, :self.dimension] = np.reshape(coordinates_a, (2, self.dimension))
                x[j, i, :, self.dimension:] = np.reshape(coordinates_b, (2, self.dimension))
        
        return np.array([x, x])
    
    def perform_augmentation(self, sequence_id, coordinates_1, coordinates_2):
        """
        Augments coordinates for training
        Arguments:
        coordinates -- all x coordinates followed by all y coordinates
        """
        sequence_boundary = self.sequence_boundaries[sequence_id]
            
        result = augment_pair_coordinates(coordinates_1, 
                                          coordinates_2,
                                          randomize_positions=self.randomize_positions,
                                          mean_scale_factor=self.mean_scale_factor,
                                          p_horizontal_flip=self.p_horizontal_flip
                                    )
        return result
    
    def on_epoch_end(self):
        """
        Called on epoch end. If randomized is set to true, it shuffles sequence IDs for the next pass
        """
        if self.randomized:
            self.ids = np.random.permutation(self.ids)
        self.index = 0
        self.length = None
        self.__len__()
