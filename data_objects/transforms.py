import numpy as np
import random

class Normalize(object):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std

    def __call__(self, input):
        return (input - self.mean) / self.std


class TimeReverse(object):
    def __init__(self, p=0.5):
        super(TimeReverse, self).__init__()
        self.p = p

    def __call__(self, input):
        if random.random() < self.p:
            return np.flip(input, axis=0).copy()
        return input


def generate_test_sequence(feature, partial_n_frames, shift=None):
    while feature.shape[0] <= partial_n_frames:
        feature = np.repeat(feature, 2, axis=0)
    if shift is None:
        shift = partial_n_frames // 2
    test_sequence = []
    start = 0
    while start + partial_n_frames <= feature.shape[0]:
        test_sequence.append(feature[start: start + partial_n_frames])
        start += shift
    test_sequence = np.stack(test_sequence, axis=0)
    return test_sequence