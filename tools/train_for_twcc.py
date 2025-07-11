# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil

import numpy as np
import random

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

import _init_paths
from config import cfg
from config import update_config

from core.loss import JointsMSELoss
from core.loss import JointsOHKMMSELoss
from core.function import train
from core.function import validate
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.utils import get_model_summary

import dataset
import models


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly
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
        cfg, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
    
    seed = 22
    torch.manual_seed(seed)       # 設置 CPU 隨機種子
    np.random.seed(seed)          # 設置 NumPy 隨機種子
    random.seed(seed)             # 設置 Python 內建 random 模塊的隨機種子
    torch.cuda.manual_seed(seed)  # 設置當前 GPU 的隨機種子
    torch.cuda.manual_seed_all(seed)  # 設置所有 GPU 的隨機種子

    model, model_fine = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=True)

    # copy model file
    this_dir = os.path.dirname(__file__)
    shutil.copy2(
        os.path.join(this_dir, '../lib/models', cfg.MODEL.NAME + '.py'),
        final_output_dir)

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    dump_input = torch.rand(
        (1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0])
    )
    writer_dict['writer'].add_graph(model, (dump_input, ))
    logger.info(get_model_summary(model, dump_input))

    dump_input_fine = torch.rand(1, cfg.MODEL.NUM_JOINTS, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0])  #gaidong
    writer_dict['writer'].add_graph(model_fine, (dump_input_fine, ))
    logger.info(get_model_summary(model_fine, dump_input_fine))

    #model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
    #model_fine = torch.nn.DataParallel(model_fine,device_ids=[0]).cuda()         # gaidong
    model = torch.nn.DataParallel(model, device_ids=[0,1,2,3]).cuda()
    model_fine = torch.nn.DataParallel(model_fine, device_ids=[0,1,2,3]).cuda()         # gaidong
    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()
    criterion_fine = JointsOHKMMSELoss(False).cuda()
    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True,
        transforms.Compose([
            transforms.ToTensor(),#把[0,255]形状为[H,W,C]的图片转化为[1,1.0]形状为[C,H,W]的torch.FloatTensor
            normalize,
        ])
    )
    train_test_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, False,
        transforms.Compose([
            transforms.ToTensor(),#把[0,255]形状为[H,W,C]的图片转化为[1,1.0]形状为[C,H,W]的torch.FloatTensor
            normalize,
        ])
    )
    valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    train_test_loader = torch.utils.data.DataLoader(
        train_test_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    best_perf = 99999 # for distance metric
    best_model = False
    last_epoch = -1
    optimizer=torch.optim.Adam([{"params":model.parameters()}, {"params":model_fine.parameters()}], lr=cfg.TRAIN.LR) #gaidong
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth'
    )

    #Auto Resume
    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        # 添加这一段
        if 'state_dict_fine' in checkpoint:
            model_fine.load_state_dict(checkpoint['state_dict_fine'])
        else:
            logger.warning("=> No state_dict_fine found in checkpoint, model_fine will use random initialization")
    
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=last_epoch
    )

    # 訓練開始前釋放記憶體
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        # train for one epoch
        train(cfg, train_loader, model, model_fine, criterion, criterion_fine,optimizer, epoch,
              final_output_dir, tb_log_dir, writer_dict)
        
        # 更新學習率（應該在 optimizer.step() 之後）
        lr_scheduler.step()

        print('lr: ', lr_scheduler.get_last_lr())  # 修正 get_lr() 問題

        # evaluate on training set
        perf_indicator = validate(
            cfg, train_test_loader, train_test_dataset, model, model_fine,criterion, criterion_fine,
            final_output_dir, tb_log_dir, writer_dict, data_type='train'
        )
        
        # evaluate on validation set
        perf_indicator = validate(
            cfg, valid_loader, valid_dataset, model, model_fine,criterion, criterion_fine,
            final_output_dir, tb_log_dir, writer_dict
        )
        if perf_indicator <= best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        
        # save checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'model': cfg.MODEL.NAME,
            'state_dict': model.state_dict(),
            'state_dict_fine': model_fine.state_dict(),  # 添加这一行
            'best_state_dict': model.module.state_dict(),
            'best_state_dict_fine': model_fine.module.state_dict(),  # 添加这一行
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir)

    final_model_state_file = os.path.join(
        final_output_dir, 'final_state.pth'
    )
    logger.info('=> saving final model state to {}'.format(
        final_model_state_file)
    )
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
