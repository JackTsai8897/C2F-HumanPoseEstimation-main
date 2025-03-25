# import torch
# print(torch.cuda.is_available())

# print(torch.__version__)
# print(torch.cuda.get_device_name(0))

import os.path as osp
import sys

import _init_paths
from config import cfg
from config import update_config

import dataset
import torchvision.transforms as transforms

# Update config from experiments
cfg.defrost()
cfg.merge_from_file("../experiments/mydataset/hrnet/new_w48_512x224_adam_lr1e-3.yaml")
# cfg.merge_from_list("../experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml")
cfg.DATASET.ROOT = os.path.join(
        "..", cfg.DATA_DIR, cfg.DATASET.ROOT
    )
cfg.DATASET.SCALE_FACTOR = 0.0
cfg.DATASET.ROT_FACTOR = 0
cfg.DATASET.PROB_HALF_BODY = 0.0
cfg.DATASET.NUM_JOINTS_HALF_BODY = 0
cfg.DATASET.PADDING_IMAGE_TO_BLACK_WITHOUT_BBOX = True # padding image to black without bbox
cfg.DATASET.BBOX_PADDING_FACTOR = -0.1 # extend padding area

cfg.freeze()

# cfg.MODEL.IMAGE_SIZE = [512, 224]
# cfg.MODEL.HEATMAP_SIZE = [128, 56]
# cfg.DATASET.ROT_FACTOR = 0
normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
# train_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
#         cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True,
#         transforms.Compose([
#             transforms.ToTensor(),#把[0,255]形状为[H,W,C]的图片转化为[1,1.0]形状为[C,H,W]的torch.FloatTensor
#             normalize,
#         ])
#     )
valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )

# input, target, target_weight, meta = train_dataset[0]
input, target, target_weight, meta = valid_dataset[40]

import torch
import numpy as np
import cv2

def tensor_to_cv2_image(tensor):
    """
    將 PyTorch 張量轉換為 OpenCV 可儲存的影像格式
    
    參數:
        tensor: 形狀為 [3, H, W] 的 PyTorch 張量，值範圍在 [0, 1] 或 [-1, 1]
        
    返回:
        OpenCV 格式的影像 (numpy 陣列，形狀為 [H, W, 3]，BGR 順序，值範圍在 [0, 255])
    """
    # 1. 確保張量在 CPU 上並轉換為 numpy 陣列
    if tensor.device.type != 'cpu':
        tensor = tensor.cpu()
    
    # 2. 將張量轉換為 numpy 陣列
    img = tensor.detach().numpy()
    
    # 3. 將通道從第一維移到最後一維 (CHW -> HWC)
    img = np.transpose(img, (1, 2, 0))
    
    # 4. 如果值範圍在 [-1, 1]，轉換到 [0, 1]
    if img.min() < 0:
        img = (img + 1) / 2
    
    # 5. 確保值範圍在 [0, 1]
    img = np.clip(img, 0, 1)
    
    # 6. 將值範圍從 [0, 1] 轉換到 [0, 255]
    img = (img * 255).astype(np.uint8)
    
    # 7. 將 RGB 轉換為 BGR（PyTorch 通常是 RGB，而 OpenCV 使用 BGR）
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    return img_bgr
def heatmaps_to_colored_image(heatmaps, colormap=cv2.COLORMAP_JET, normalize=True, combine_method='max'):
    """
    將多通道熱力圖張量轉換為可視化的彩色影像
    
    參數:
        heatmaps: 形狀為 [C, H, W] 的 PyTorch 張量，表示多個熱力圖
        colormap: OpenCV 顏色映射，默認為 COLORMAP_JET
        normalize: 是否對每個熱力圖進行歸一化
        combine_method: 如何組合多個熱力圖，可選 'max', 'mean', 'sum', 或指定通道索引
        
    返回:
        彩色熱力圖影像 (numpy 陣列，形狀為 [H, W, 3]，BGR 順序，值範圍在 [0, 255])
    """
    # 1. 確保張量在 CPU 上並轉換為 numpy 陣列
    if heatmaps.device.type != 'cpu':
        heatmaps = heatmaps.cpu()
    
    heatmaps_np = heatmaps.detach().numpy()
    
    # 2. 根據指定方法組合多個熱力圖
    if isinstance(combine_method, int) and 0 <= combine_method < heatmaps_np.shape[0]:
        # 使用指定通道
        combined_heatmap = heatmaps_np[combine_method]
    elif combine_method == 'max':
        # 取每個位置的最大值
        combined_heatmap = np.max(heatmaps_np, axis=0)
    elif combine_method == 'mean':
        # 取平均值
        combined_heatmap = np.mean(heatmaps_np, axis=0)
    elif combine_method == 'sum':
        # 取總和
        combined_heatmap = np.sum(heatmaps_np, axis=0)
    else:
        raise ValueError("combine_method 必須是 'max', 'mean', 'sum' 或有效的通道索引")
    
    # 3. 歸一化熱力圖到 [0, 1] 範圍
    if normalize:
        min_val = combined_heatmap.min()
        max_val = combined_heatmap.max()
        if max_val > min_val:
            combined_heatmap = (combined_heatmap - min_val) / (max_val - min_val)
        else:
            combined_heatmap = np.zeros_like(combined_heatmap)
    
    # 4. 轉換到 [0, 255] 範圍並轉為 uint8
    heatmap_uint8 = (combined_heatmap * 255).astype(np.uint8)
    
    # 5. 應用顏色映射
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
    
    return colored_heatmap

def visualize_all_heatmaps(heatmaps, colormap=cv2.COLORMAP_JET, normalize=True):
    """
    將多通道熱力圖張量轉換為多個彩色影像並排顯示
    
    參數:
        heatmaps: 形狀為 [C, H, W] 的 PyTorch 張量，表示多個熱力圖
        colormap: OpenCV 顏色映射，默認為 COLORMAP_JET
        normalize: 是否對每個熱力圖進行歸一化
        
    返回:
        包含所有熱力圖的網格影像 (numpy 陣列，BGR 順序，值範圍在 [0, 255])
    """
    # 1. 確保張量在 CPU 上並轉換為 numpy 陣列
    if heatmaps.device.type != 'cpu':
        heatmaps = heatmaps.cpu()
    
    heatmaps_np = heatmaps.detach().numpy()
    num_channels = heatmaps_np.shape[0]
    
    # 2. 計算網格布局
    grid_size = int(np.ceil(np.sqrt(num_channels)))
    
    # 3. 創建網格圖像
    h, w = heatmaps_np.shape[1], heatmaps_np.shape[2]
    grid_image = np.zeros((h * grid_size, w * grid_size, 3), dtype=np.uint8)
    
    # 4. 為每個通道生成熱力圖並放入網格
    for i in range(num_channels):
        row = i // grid_size
        col = i % grid_size
        
        # 獲取單個通道的熱力圖
        single_heatmap = heatmaps_np[i]
        
        # 歸一化
        if normalize:
            min_val = single_heatmap.min()
            max_val = single_heatmap.max()
            if max_val > min_val:
                single_heatmap = (single_heatmap - min_val) / (max_val - min_val)
            else:
                single_heatmap = np.zeros_like(single_heatmap)
        
        # 轉換到 [0, 255] 範圍並轉為 uint8
        heatmap_uint8 = (single_heatmap * 255).astype(np.uint8)
        
        # 應用顏色映射
        colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
        
        # 放入網格
        grid_image[row*h:(row+1)*h, col*w:(col+1)*w] = colored_heatmap
    
    return grid_image

import os
outputdir = './output_images'
# 轉換張量為 OpenCV 影像
cv2_image = tensor_to_cv2_image(input)

# 儲存影像
# cv2.imwrite('./{}'.format(meta["image"].split("\\")[-1]), input)
cv2.imwrite(os.path.join(outputdir, 'output_image.jpg'), cv2_image)

heatmaps=target
# 方法 1: 生成組合熱力圖
combined_heatmap = heatmaps_to_colored_image(heatmaps, colormap=cv2.COLORMAP_JET)
cv2.imwrite(os.path.join(outputdir,'combined_heatmap.jpg'), combined_heatmap)

# 方法 2: 生成所有通道的熱力圖網格
all_heatmaps = visualize_all_heatmaps(heatmaps)
cv2.imwrite(os.path.join(outputdir,'all_heatmaps.jpg'), all_heatmaps)

# 方法 3: 顯示特定通道的熱力圖
for i in range(6):
    single_heatmap = heatmaps_to_colored_image(heatmaps, combine_method=i)
    cv2.imwrite(os.path.join(outputdir, f'heatmap_channel_{i}.jpg'), single_heatmap)



def denormalize_tensor(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    將標準化後的張量反標準化回原始圖像數值範圍
    
    參數:
        tensor: 形狀為 [C, H, W] 的 PyTorch 張量，已經過標準化
        mean: 標準化時使用的均值
        std: 標準化時使用的標準差
        
    返回:
        反標準化後的張量，值範圍在 [0, 1]
    """
    # 確保輸入是張量
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor)
        
    # 創建與輸入張量相同形狀的均值和標準差張量
    mean = torch.tensor(mean, device=tensor.device).view(-1, 1, 1)
    std = torch.tensor(std, device=tensor.device).view(-1, 1, 1)
    
    # 反標準化: pixel = (normalized_pixel * std) + mean
    denormalized = tensor * std + mean
    
    # 確保值在 [0, 1] 範圍內
    denormalized = torch.clamp(denormalized, 0, 1)
    
    return denormalized

def tensor_to_image(tensor, denormalize=True, to_uint8=True, 
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    將 PyTorch 張量轉換為可顯示的圖像
    
    參數:
        tensor: 形狀為 [C, H, W] 的 PyTorch 張量
        denormalize: 是否需要反標準化
        to_uint8: 是否轉換為 uint8 類型 (0-255 範圍)
        mean: 標準化時使用的均值
        std: 標準化時使用的標準差
        
    返回:
        numpy 陣列，形狀為 [H, W, C]，可用於 OpenCV 顯示或保存
    """
    # 確保張量在 CPU 上
    if tensor.device.type != 'cpu':
        tensor = tensor.cpu()
    
    # 分離張量，避免計算梯度
    tensor = tensor.detach()
    
    # 如果需要反標準化
    if denormalize:
        tensor = denormalize_tensor(tensor, mean, std)
    
    # 將張量轉換為 numpy 陣列
    img_np = tensor.numpy()
    
    # 調整通道順序：從 [C, H, W] 到 [H, W, C]
    img_np = np.transpose(img_np, (1, 2, 0))
    
    # 如果需要轉換為 uint8 (0-255 範圍)
    if to_uint8:
        img_np = (img_np * 255).astype(np.uint8)
    
    # 將 RGB 轉換為 BGR (適用於 OpenCV)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    return img_bgr

def save_tensor_as_image(tensor, filepath, denormalize=True, 
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    將張量保存為圖像文件
    
    參數:
        tensor: 形狀為 [C, H, W] 的 PyTorch 張量
        filepath: 保存圖像的路徑
        denormalize: 是否需要反標準化
        mean: 標準化時使用的均值
        std: 標準化時使用的標準差
    """
    img = tensor_to_image(tensor, denormalize, True, mean, std)
    cv2.imwrite(filepath, img)

def display_tensor_as_image(tensor, denormalize=True, 
                           mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    顯示張量為圖像 (使用 OpenCV)
    
    參數:
        tensor: 形狀為 [C, H, W] 的 PyTorch 張量
        denormalize: 是否需要反標準化
        mean: 標準化時使用的均值
        std: 標準化時使用的標準差
    """
    img = tensor_to_image(tensor, denormalize, True, mean, std)
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 保存為圖像
save_tensor_as_image(input, os.path.join(outputdir,'denormalized_image.jpg'))
    
# 顯示圖像
# display_tensor_as_image(normalized_tensor)