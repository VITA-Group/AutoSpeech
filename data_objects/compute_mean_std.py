from data_objects.speaker import Speaker
import numpy as np

def compute_mean_std(dataset_dir, output_path_mean, output_path_std):
    print("Computing mean std...")
    speaker_dirs = [f for f in dataset_dir.glob("*") if f.is_dir()]
    if len(speaker_dirs) == 0:
        raise Exception("No speakers found. Make sure you are pointing to the directory "
                        "containing all preprocessed speaker directories.")
    speakers = [Speaker(speaker_dir) for speaker_dir in speaker_dirs]

    sources = []
    for speaker in speakers:
        sources.extend(speaker.sources)

    sumx = np.zeros(257, dtype=np.float32)
    sumx2 = np.zeros(257, dtype=np.float32)
    count = 0
    n = len(sources)
    for i, source in enumerate(sources):
        feature = np.load(source[0].joinpath(source[1]))
        sumx += feature.sum(axis=0)
        sumx2 += (feature * feature).sum(axis=0)
        count += feature.shape[0]

    mean = sumx / count
    std = np.sqrt(sumx2 / count - mean * mean)

    mean = mean.astype(np.float32)
    std = std.astype(np.float32)

    np.save(output_path_mean, mean)
    np.save(output_path_std, std)