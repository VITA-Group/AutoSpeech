# -*- coding: utf-8 -*-
# @Date    : 2019-08-09
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import shutil
import os
from tensorboardX import SummaryWriter
from tqdm import tqdm
from pathlib import Path

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

from config import cfg, update_config
from utils import set_path, create_logger, save_checkpoint
from data_objects.DeepSpeakerDataset import DeepSpeakerDataset
from functions import train, validate_identification
from architect import Architect
from loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from spaces import primitives_1, primitives_2, primitives_3
from models.model_search import Network


def parse_args():
    parser = argparse.ArgumentParser(description='Train energy network')
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
                        default=None)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    update_config(cfg, args)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    # Set the random seed manually for reproducibility.
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed_all(cfg.SEED)

    # Loss
    criterion = CrossEntropyLoss(cfg.MODEL.NUM_CLASSES).cuda()

    # model and optimizer
    model = Network(cfg.MODEL.INIT_CHANNELS, cfg.MODEL.NUM_CLASSES, cfg.MODEL.LAYERS, criterion, primitives_2,
                    drop_path_prob=cfg.TRAIN.DROPPATH_PROB)
    model = model.cuda()

    # weight params
    arch_params = list(map(id, model.arch_parameters()))
    weight_params = filter(lambda p: id(p) not in arch_params,
                           model.parameters())

    # Optimizer
    optimizer = optim.Adam(
        weight_params,
        lr=cfg.TRAIN.LR
    )

    # resume && make log dir and logger
    if args.load_path and os.path.exists(args.load_path):
        checkpoint_file = os.path.join(args.load_path, 'Model', 'checkpoint_best.pth')
        assert os.path.exists(checkpoint_file)
        checkpoint = torch.load(checkpoint_file)

        # load checkpoint
        begin_epoch = checkpoint['epoch']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        best_acc1 = checkpoint['best_acc1']
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.path_helper = checkpoint['path_helper']

        logger = create_logger(args.path_helper['log_path'])
        logger.info("=> loaded checkpoint '{}'".format(checkpoint_file))
    else:
        exp_name = args.cfg.split('/')[-1].split('.')[0]
        args.path_helper = set_path('logs_search', exp_name)
        logger = create_logger(args.path_helper['log_path'])
        begin_epoch = cfg.TRAIN.BEGIN_EPOCH
        best_acc1 = 0.0
        last_epoch = -1

    logger.info(args)
    logger.info(cfg)

    # copy model file
    this_dir = os.path.dirname(__file__)
    shutil.copy2(
        os.path.join(this_dir, 'models', cfg.MODEL.NAME + '.py'),
        args.path_helper['ckpt_path'])

    # dataloader
    train_dataset = DeepSpeakerDataset(
        Path(cfg.DATASET.DATA_DIR), cfg.DATASET.SUB_DIR, cfg.DATASET.PARTIAL_N_FRAMES, 'train')
    val_dataset = DeepSpeakerDataset(
        Path(cfg.DATASET.DATA_DIR), cfg.DATASET.SUB_DIR, cfg.DATASET.PARTIAL_N_FRAMES, 'val')
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.DATASET.NUM_WORKERS,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.DATASET.NUM_WORKERS,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )
    test_dataset = DeepSpeakerDataset(
        Path(cfg.DATASET.DATA_DIR), cfg.DATASET.SUB_DIR, cfg.DATASET.PARTIAL_N_FRAMES, 'test', is_test=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=1,
        num_workers=cfg.DATASET.NUM_WORKERS,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )

    # training setting
    writer_dict = {
        'writer': SummaryWriter(args.path_helper['log_path']),
        'train_global_steps': begin_epoch * len(train_loader),
        'valid_global_steps': begin_epoch // cfg.VAL_FREQ,
    }

    # training loop
    architect = Architect(model, cfg)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, cfg.TRAIN.END_EPOCH, cfg.TRAIN.LR_MIN,
        last_epoch=last_epoch
    )

    for epoch in tqdm(range(begin_epoch, cfg.TRAIN.END_EPOCH), desc='search progress'):
        model.train()

        genotype = model.genotype()
        logger.info('genotype = %s', genotype)

        if cfg.TRAIN.DROPPATH_PROB != 0:
            model.drop_path_prob = cfg.TRAIN.DROPPATH_PROB * epoch / (cfg.TRAIN.END_EPOCH - 1)

        train(cfg, model, optimizer, train_loader, val_loader, criterion, architect, epoch, writer_dict)

        if epoch % cfg.VAL_FREQ == 0:
            # get threshold and evaluate on validation set
            acc = validate_identification(cfg, model, test_loader, criterion)

            # remember best acc@1 and save checkpoint
            is_best = acc > best_acc1
            best_acc1 = max(acc, best_acc1)

            # save
            logger.info('=> saving checkpoint to {}'.format(args.path_helper['ckpt_path']))
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                'arch': model.arch_parameters(),
                'genotype': genotype,
                'path_helper': args.path_helper
            }, is_best, args.path_helper['ckpt_path'], 'checkpoint_{}.pth'.format(epoch))

        lr_scheduler.step(epoch)



if __name__ == '__main__':
    main()
