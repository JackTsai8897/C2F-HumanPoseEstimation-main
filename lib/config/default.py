
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN


_C = CN()

_C.OUTPUT_DIR = ''
_C.LOG_DIR = ''
_C.DATA_DIR = ''
_C.GPUS = (0,)
_C.WORKERS = 4
_C.PRINT_FREQ = 20
_C.AUTO_RESUME = False
_C.PIN_MEMORY = True
_C.RANK = 0

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'pose_hrnet'
_C.MODEL.INIT_WEIGHTS = True
_C.MODEL.PRETRAINED = ''
_C.MODEL.NUM_JOINTS = 6
_C.MODEL.TAG_PER_JOINT = True
_C.MODEL.TARGET_TYPE = 'gaussian'
_C.MODEL.IMAGE_SIZE = [384, 288]  # width * height, ex: 192 * 256
_C.MODEL.HEATMAP_SIZE = [96, 72]  # width * height, ex: 24 * 32
_C.MODEL.SIGMA = 3
_C.MODEL.EXTRA = CN(new_allowed=True)
# Transformer
_C.MODEL.BOTTLENECK_NUM = 0
_C.MODEL.DIM_MODEL = 256
_C.MODEL.DIM_FEEDFORWARD = 512
_C.MODEL.ENCODER_LAYERS = 6
_C.MODEL.N_HEAD = 8
_C.MODEL.ATTENTION_ACTIVATION = 'relu'
_C.MODEL.POS_EMBEDDING = 'learnable'
_C.MODEL.INTERMEDIATE_SUP = False
_C.MODEL.PE_ONLY_AT_BEGIN = False
#### 

_C.LOSS = CN()
_C.LOSS.USE_OHKM = False
_C.LOSS.TOPK = 8
_C.LOSS.USE_TARGET_WEIGHT = True
_C.LOSS.USE_DIFFERENT_JOINTS_WEIGHT = False

# DATASET related params
_C.DATASET = CN()
_C.DATASET.ROOT = '../data/mydataset/'
_C.DATASET.DATASET = 'mydataset'
_C.DATASET.TRAIN_SET = 'train'
_C.DATASET.TEST_SET = 'val'
_C.DATASET.DATA_FORMAT = 'png'
_C.DATASET.HYBRID_JOINTS_TYPE = ''
_C.DATASET.SELECT_DATA = False
_C.DATASET.HAND_ORIENTATION = ''

# training data augmentation
_C.DATASET.FLIP = False
_C.DATASET.SCALE_FACTOR = 0.0
_C.DATASET.ROT_FACTOR = 0
_C.DATASET.PROB_HALF_BODY = 0.0
_C.DATASET.NUM_JOINTS_HALF_BODY = 0
_C.DATASET.COLOR_RGB = False
_C.DATASET.UPSAMPLE_FACTOR = 0
_C.DATASET.PADDING_IMAGE_TO_BLACK_WITHOUT_BBOX = False
_C.DATASET.BBOX_PADDING_FACTOR = 0.0

# train
_C.TRAIN = CN()

_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP = [90, 110]
_C.TRAIN.LR = 0.001
_C.TRAIN.LR_END = 0.00001

_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 0.0001
_C.TRAIN.NESTEROV = False
_C.TRAIN.GAMMA1 = 0.99
_C.TRAIN.GAMMA2 = 0.0

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 140

_C.TRAIN.RESUME = False
_C.TRAIN.CHECKPOINT = ''

_C.TRAIN.BATCH_SIZE_PER_GPU = 32
_C.TRAIN.SHUFFLE = True

# testing
_C.TEST = CN()

# darkpose
_C.TEST.BLUR_KERNEL = 3 

# size of images for each device
_C.TEST.BATCH_SIZE_PER_GPU = 32
# Test Model Epoch
_C.TEST.FLIP_TEST = False
_C.TEST.POST_PROCESS = False
_C.TEST.SHIFT_HEATMAP = False

_C.TEST.USE_GT_BBOX = False

# nms
_C.TEST.IMAGE_THRE = 0.1
_C.TEST.NMS_THRE = 0.6
_C.TEST.SOFT_NMS = False
_C.TEST.OKS_THRE = 0.5
_C.TEST.IN_VIS_THRE = 0.0
_C.TEST.COCO_BBOX_FILE = ''
_C.TEST.BBOX_THRE = 1.0
_C.TEST.MODEL_FILE = ''

# debug
_C.DEBUG = CN()
_C.DEBUG.DEBUG = False
_C.DEBUG.SAVE_BATCH_IMAGES_GT = False
_C.DEBUG.SAVE_BATCH_IMAGES_PRED = False
_C.DEBUG.SAVE_HEATMAPS_GT = False
_C.DEBUG.SAVE_HEATMAPS_PRED = False


def update_config(cfg, args):
    cfg.defrost()
    # 对于不同的模型配置，你有不同的超参设置，所以你可以使用yaml文件来管理不同的configs，
    # 然后使用merge_from_file()这个方法，这个会比较每个模型特有的config和默认参数的区别，
    # 会将默认参数与特定参数不同的部分，用特定参数覆盖
    cfg.merge_from_file(args.cfg)
    # 对于不同的模型配置，你有不同的超参设置，所以你可以使用列表来管理不同的configs，
    # 然后使用merge_from_list()这个方法，这个会比较每个模型特有的config和默认参数的区别，
    # 会将默认参数与特定参数不同的部分，用特定参数覆盖
    cfg.merge_from_list(args.opts)

    if args.modelDir:
        cfg.OUTPUT_DIR = args.modelDir

    if args.logDir:
        cfg.LOG_DIR = args.logDir

    if args.dataDir:
        cfg.DATA_DIR = args.dataDir

    cfg.DATASET.ROOT = os.path.join(
        cfg.DATA_DIR, cfg.DATASET.ROOT
    )

    cfg.MODEL.PRETRAINED = os.path.join(
        cfg.DATA_DIR, cfg.MODEL.PRETRAINED
    )

    if cfg.TEST.MODEL_FILE:
        cfg.TEST.MODEL_FILE = os.path.join(
            cfg.DATA_DIR, cfg.TEST.MODEL_FILE
        )

    cfg.freeze()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)

