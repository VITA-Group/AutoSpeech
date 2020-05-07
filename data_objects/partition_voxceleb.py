import os


def partition_voxceleb(feature_root, split_txt_path):
    print("partitioning VoxCeleb...")
    with open(split_txt_path, 'r') as f:
        split_txt = f.readlines()
    train_set = []
    val_set = []
    test_set = []
    for line in split_txt:
        items = line.strip().split()
        if items[0] == '3':
            test_set.append(items[1])
        elif items[0] == '2':
            val_set.append(items[1])
        else:
            train_set.append(items[1])

    speakers = os.listdir(feature_root)

    for speaker in speakers:
        speaker_dir = os.path.join(feature_root, speaker)
        if not os.path.isdir(speaker_dir):
            continue
        with open(os.path.join(speaker_dir, '_sources.txt'), 'r') as f:
            speaker_files = f.readlines()

        train = []
        val = []
        test = []
        for line in speaker_files:
            address = line.strip().split(',')[1]
            fname = os.path.join(*address.split('/')[-3:])
            if fname in test_set:
                test.append(line)
            elif fname in val_set:
                val.append(line)
            elif fname in train_set:
                train.append(line)
            else:
                print('file not in either train or test set')

        with open(os.path.join(speaker_dir, '_sources_train.txt'), 'w') as f:
            f.writelines('%s' % line for line in train)
        with open(os.path.join(speaker_dir, '_sources_val.txt'), 'w') as f:
            f.writelines('%s' % line for line in val)
        with open(os.path.join(speaker_dir, '_sources_test.txt'), 'w') as f:
            f.writelines('%s' % line for line in test)
