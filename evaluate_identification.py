from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from models.model import Network
from models import resnet
from config import cfg, update_config
from utils import create_logger, Genotype
from data_objects.DeepSpeakerDataset import DeepSpeakerDataset
from functions import validate_identification


def parse_args():
    parser = argparse.ArgumentParser(description='Train autospeech network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--load_path',
                        help="The path to resumed dir",
                        required=True,
                        default=None)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    update_config(cfg, args)
    if args.load_path is None:
        raise AttributeError("Please specify load path.")

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    # Set the random seed manually for reproducibility.
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed_all(cfg.SEED)

    # model and optimizer
    if cfg.MODEL.NAME == 'model':
        if args.load_path and os.path.exists(args.load_path):
            checkpoint = torch.load(args.load_path)
            genotype = checkpoint['genotype']
        else:
            raise AssertionError('Please specify the model to evaluate')
        model = Network(cfg.MODEL.INIT_CHANNELS, cfg.MODEL.NUM_CLASSES, cfg.MODEL.LAYERS, genotype)
        model.drop_path_prob = 0.0
    else:
        model = eval('resnet.{}(num_classes={})'.format(cfg.MODEL.NAME, cfg.MODEL.NUM_CLASSES))
    model = model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()

    # resume && make log dir and logger
    if args.load_path and os.path.exists(args.load_path):
        checkpoint = torch.load(args.load_path)

        # load checkpoint
        model.load_state_dict(checkpoint['state_dict'])
        args.path_helper = checkpoint['path_helper']

        logger = create_logger(os.path.dirname(args.load_path))
        logger.info("=> loaded checkpoint '{}'".format(args.load_path))
    else:
        raise AssertionError('Please specify the model to evaluate')
    logger.info(args)
    logger.info(cfg)

    # dataloader
    test_dataset_identification = DeepSpeakerDataset(
        Path(cfg.DATASET.DATA_DIR), cfg.DATASET.SUB_DIR, cfg.DATASET.PARTIAL_N_FRAMES, 'test', is_test=True)


    test_loader_identification = torch.utils.data.DataLoader(
        dataset=test_dataset_identification,
        batch_size=1,
        num_workers=cfg.DATASET.NUM_WORKERS,
        pin_memory=True,
        shuffle=False,
        drop_last=True,
    )

    validate_identification(cfg, model, test_loader_identification, criterion)



if __name__ == '__main__':
    main()
