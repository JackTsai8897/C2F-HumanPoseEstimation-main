# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.transforms import get_affine_transform
from utils.transforms import affine_transform

logger = logging.getLogger(__name__)


class HandKeypointsDataset(Dataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        self.num_joints = 0
        self.pixel_std = 200
        self.flip_pairs = []
        self.parent_ids = []

        self.is_train = is_train
        self.root = root
        self.image_set = image_set

        self.output_path = cfg.OUTPUT_DIR
        self.data_format = cfg.DATASET.DATA_FORMAT

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.color_rgb = cfg.DATASET.COLOR_RGB

        self.target_type = cfg.MODEL.TARGET_TYPE
        self.image_size = np.array(cfg.MODEL.IMAGE_SIZE)
        self.heatmap_size = np.array(cfg.MODEL.HEATMAP_SIZE)
        self.sigma = cfg.MODEL.SIGMA
        self.use_different_joints_weight = cfg.LOSS.USE_DIFFERENT_JOINTS_WEIGHT
        self.joints_weight = 1

        self.transform = transform
        self.db = []

    def _get_db(self):
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError

    def __len__(self,):
        return len(self.db)

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        image_file = db_rec['image']
        filename = db_rec['filename'] if 'filename' in db_rec else ''
        imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''

        if self.data_format == 'zip':
            from utils import zipreader
            data_numpy = zipreader.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
        else:
            data_numpy = cv2.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )

        if self.color_rgb:
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

        if data_numpy is None:
            logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))

        joints = db_rec['joints_3d']
        joints_vis = db_rec['joints_3d_vis']

        c = db_rec['center']
        s = db_rec['scale']
        score = db_rec['score'] if 'score' in db_rec else 1
        r = 0

        if self.is_train:
            self.add_gaussian_noise_rand(data_numpy)
            self.add_color_jittering(data_numpy)

            # sf = self.scale_factor
            rf = self.rotation_factor
            # s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
            r = r = random.uniform(-rf, rf)

        trans = get_affine_transform(c, s, r, self.image_size)
        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        if self.transform:
            input = self.transform(input)

        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

        target, target_weight = self.generate_target(joints, joints_vis)

        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)

        meta = {
            'image': image_file,
            'filename': filename,
            'imgnum': imgnum,
            'joints': joints,
            'joints_vis': joints_vis,
            'center': c,
            'scale': s,
            'rotation': r,
            'score': score
        }

        return input, target, target_weight, meta

    def select_data(self, db):
        db_selected = []
        for rec in db:
            num_vis = 0
            joints_x = 0.0
            joints_y = 0.0
            for joint, joint_vis in zip(
                    rec['joints_3d'], rec['joints_3d_vis']):
                if joint_vis[0] <= 0:
                    continue
                num_vis += 1

                joints_x += joint[0]
                joints_y += joint[1]
            if num_vis == 0:
                continue

            joints_x, joints_y = joints_x / num_vis, joints_y / num_vis

            area = rec['scale'][0] * rec['scale'][1] * (self.pixel_std**2)
            joints_center = np.array([joints_x, joints_y])
            bbox_center = np.array(rec['center'])
            diff_norm2 = np.linalg.norm((joints_center-bbox_center), 2)
            ks = np.exp(-1.0*(diff_norm2**2) / ((0.2)**2*2.0*area))

            metric = (0.2 / 16) * num_vis + 0.45 - 0.2 / 16
            if ks > metric:
                db_selected.append(rec)

        logger.info('=> num db: {}'.format(len(db)))
        logger.info('=> num selected db: {}'.format(len(db_selected)))
        return db_selected

    def generate_target(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros((self.num_joints,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = self.sigma * 3

            for joint_id in range(self.num_joints):
                feat_stride = self.image_size / self.heatmap_size
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target, target_weight
    
    def add_gaussian_noise(self, image, mean=0, std=5):
        """
        添加高斯噪聲到圖像
        
        Args:
            image: 輸入圖像 (numpy array)
            mean: 高斯分佈的均值
            std: 高斯分佈的標準差
            
        Returns:
            添加噪聲後的圖像
        """
        # 確保輸入是浮點型
        image_float = image.astype(np.float32)
        
        # 生成與圖像相同形狀的高斯噪聲
        noise = np.random.normal(mean, std, image.shape)
        
        # 將噪聲添加到圖像
        noisy_image = image_float + noise
        
        # 裁剪值到有效範圍 [0, 255]
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        
        return noisy_image
    
    def add_gaussian_noise_rand(self, image, mean=0, std_range=(0, 50)):
        """
        添加高斯噪聲到圖像，標準差在指定範圍內隨機生成
        
        Args:
            image: 輸入圖像 (numpy array)
            mean: 高斯分佈的均值
            std_range: 高斯分佈標準差的範圍，例如(0, 50)表示標準差在0到50之間隨機選擇
            
        Returns:
            添加噪聲後的圖像
        """
        # 確保輸入是浮點型
        image_float = image.astype(np.float32)
        
        # 在指定範圍內隨機生成標準差
        std = random.uniform(std_range[0], std_range[1])
        
        # 生成與圖像相同形狀的高斯噪聲
        noise = np.random.normal(mean, std, image.shape)
        
        # 將噪聲添加到圖像
        noisy_image = image_float + noise
        
        # 裁剪值到有效範圍 [0, 255]
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        
        return noisy_image
    
    def add_color_jittering(self, image, strength=1):
        """
        對圖像進行顏色抖動（亮度、對比度、飽和度、色調變化）
        使用單一參數 strength 控制所有變化的強度
        
        Args:
            image: 輸入圖像 (numpy array)
            strength: 變化強度，範圍 [0, 1]，0 表示無變化，1 表示最大變化
            
        Returns:
            顏色變化後的圖像
        """
        # 確保 strength 在有效範圍內
        strength = max(0, min(1, strength))
        
        # 定義基本變化範圍
        base_brightness = 0.2
        base_contrast = 0.2
        base_saturation = 0.2
        base_hue = 0.1
        
        # 根據 strength 調整實際變化範圍
        brightness = base_brightness * strength
        contrast = base_contrast * strength
        saturation = base_saturation * strength
        hue = base_hue * strength
        
        # 轉換為 HSV 色彩空間以便於調整
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # 隨機亮度調整
        if brightness > 0:
            brightness_factor = random.uniform(max(0, 1 - brightness), 1 + brightness)
            image_hsv[:, :, 2] = image_hsv[:, :, 2] * brightness_factor
            image_hsv[:, :, 2] = np.clip(image_hsv[:, :, 2], 0, 255)
        
        # 隨機對比度調整
        if contrast > 0:
            contrast_factor = random.uniform(max(0, 1 - contrast), 1 + contrast)
            image_hsv[:, :, 2] = (image_hsv[:, :, 2] - 128) * contrast_factor + 128
            image_hsv[:, :, 2] = np.clip(image_hsv[:, :, 2], 0, 255)
        
        # 隨機飽和度調整
        if saturation > 0:
            saturation_factor = random.uniform(max(0, 1 - saturation), 1 + saturation)
            image_hsv[:, :, 1] = image_hsv[:, :, 1] * saturation_factor
            image_hsv[:, :, 1] = np.clip(image_hsv[:, :, 1], 0, 255)
        
        # 隨機色調調整
        if hue > 0:
            hue_factor = random.uniform(-hue, hue)
            image_hsv[:, :, 0] = image_hsv[:, :, 0] + hue_factor * 180
            image_hsv[:, :, 0] = np.mod(image_hsv[:, :, 0], 180)  # 色調值範圍為 [0, 180]
        
        # 轉換回 BGR 色彩空間
        image_jittered = cv2.cvtColor(image_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        return image_jittered
