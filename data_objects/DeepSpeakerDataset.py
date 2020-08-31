from __future__ import print_function


import numpy as np
import torch.utils.data as data
from data_objects.speaker import Speaker
from torchvision import transforms as T
from data_objects.transforms import Normalize, TimeReverse, generate_test_sequence


def find_classes(speakers):
    classes = list(set([speaker.name for speaker in speakers]))
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


class DeepSpeakerDataset(data.Dataset):

    def __init__(self, data_dir, sub_dir, partial_n_frames, partition=None, is_test=False):
        super(DeepSpeakerDataset, self).__init__()
        self.data_dir = data_dir
        self.root = data_dir.joinpath('feature', sub_dir)
        self.partition = partition
        self.partial_n_frames = partial_n_frames
        self.is_test = is_test

        speaker_dirs = [f for f in self.root.glob("*") if f.is_dir()]
        if len(speaker_dirs) == 0:
            raise Exception("No speakers found. Make sure you are pointing to the directory "
                            "containing all preprocessed speaker directories.")
        self.speakers = [Speaker(speaker_dir, self.partition) for speaker_dir in speaker_dirs]

        classes, class_to_idx = find_classes(self.speakers)
        sources = []
        for speaker in self.speakers:
            sources.extend(speaker.sources)
        self.features = []
        for source in sources:
            item = (source[0].joinpath(source[1]), class_to_idx[source[2]])
            self.features.append(item)
        mean = np.load(self.data_dir.joinpath('mean.npy'))
        std = np.load(self.data_dir.joinpath('std.npy'))
        self.transform = T.Compose([
            Normalize(mean, std),
            TimeReverse(),
        ])

    def load_feature(self, feature_path, speaker_id):
        feature = np.load(feature_path)
        if self.is_test:
            test_sequence = generate_test_sequence(feature, self.partial_n_frames)
            return test_sequence, speaker_id
        else:
            if feature.shape[0] <= self.partial_n_frames:
                start = 0
                while feature.shape[0] < self.partial_n_frames:
                    feature = np.repeat(feature, 2, axis=0)
            else:
                start = np.random.randint(0, feature.shape[0] - self.partial_n_frames)
            end = start + self.partial_n_frames
            return feature[start:end], speaker_id

    def __getitem__(self, index):
        feature_path, speaker_id = self.features[index]
        feature, speaker_id = self.load_feature(feature_path, speaker_id)

        if self.transform is not None:
            feature = self.transform(feature)
        return feature, speaker_id

    def __len__(self):
        return len(self.features)

