# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.loss import JointsOHKMMSELoss
from core.function import validate
from utils.utils import create_logger

import dataset
import models


def parse_args():
    parser = argparse.ArgumentParser(description='Test keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model, model_fine = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(
        cfg, is_train=False)

    
    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
        logger.warning("=> No state_dict_fine found in checkpoint, model_fine will use random initialization")
    else:
        checkpoint_file = os.path.join(
            final_output_dir, 'checkpoint.pth'
        )
        logger.info('=> loading model from {}'.format(checkpoint_file))
        
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_file)
        
        # Remove 'module.' prefix from state_dict keys
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                new_key = key[7:]  # Remove 'module.' prefix
            else:
                new_key = key
            new_state_dict[new_key] = value
        
        # Do the same for state_dict_fine if it exists
        if 'state_dict_fine' in checkpoint:
            state_dict_fine = checkpoint['state_dict_fine']
            new_state_dict_fine = {}
            for key, value in state_dict_fine.items():
                if key.startswith('module.'):
                    new_key = key[7:]  # Remove 'module.' prefix
                else:
                    new_key = key
                new_state_dict_fine[new_key] = value
            
            # Load the processed state dictionaries
            model.load_state_dict(new_state_dict)
            model_fine.load_state_dict(new_state_dict_fine)
            logger.info('=> Successfully loaded model and model_fine weights after removing module. prefix')
        else:
            # If state_dict_fine doesn't exist, just load the main model
            model.load_state_dict(new_state_dict)
            logger.warning("=> No state_dict_fine found in checkpoint, model_fine will use random initialization")

    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
    model_fine = torch.nn.DataParallel(model_fine, device_ids=[0]).cuda()

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()
    criterion_fine = JointsOHKMMSELoss(False).cuda()

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    valid_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True
    )

    # evaluate on validation set
    validate(
        cfg, valid_loader, valid_dataset, model, model_fine, criterion, criterion_fine,
        final_output_dir, tb_log_dir, wo_model_fine=True)


if __name__ == '__main__':
    main()