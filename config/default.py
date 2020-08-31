from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yacs.config import CfgNode as CN


_C = CN()

_C.PRINT_FREQ = 20
_C.VAL_FREQ = 20

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# seed
_C.SEED = 3

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'foo_net'
_C.MODEL.NUM_CLASSES = 500
_C.MODEL.LAYERS = 8
_C.MODEL.INIT_CHANNELS = 16
_C.MODEL.DROP_PATH_PROB = 0.2
_C.MODEL.PRETRAINED = False

# DATASET related params
_C.DATASET = CN()
_C.DATASET.DATA_DIR = ''
_C.DATASET.SUB_DIR = ''
_C.DATASET.TEST_DATA_DIR = ''
_C.DATASET.TEST_DATASET = ''
_C.DATASET.NUM_WORKERS = 0
_C.DATASET.PARTIAL_N_FRAMES = 32


# train
_C.TRAIN = CN()

_C.TRAIN.BATCH_SIZE = 32
_C.TRAIN.LR = 0.1
_C.TRAIN.LR_MIN = 0.001
_C.TRAIN.WD = 0.0
_C.TRAIN.BETA1 = 0.9
_C.TRAIN.BETA2 = 0.999

_C.TRAIN.ARCH_LR = 0.1
_C.TRAIN.ARCH_WD = 0.0
_C.TRAIN.ARCH_BETA1 = 0.9
_C.TRAIN.ARCH_BETA2 = 0.999

_C.TRAIN.DROPPATH_PROB = 0.2

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 140


def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    cfg.freeze()
