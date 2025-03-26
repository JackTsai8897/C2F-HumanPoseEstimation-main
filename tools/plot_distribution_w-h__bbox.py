import os.path as osp
import sys
import os

import _init_paths
from config import cfg
from config import update_config

import dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


# Update config from experiments
cfg.defrost()
cfg.merge_from_file("../experiments/mydataset/hrnet/w48_512x224_adam_lr1e-3.yaml")
# cfg.merge_from_list("../experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml")
cfg.DATASET.ROOT = os.path.join(
        "..", cfg.DATA_DIR, cfg.DATASET.ROOT
    )
cfg.DATASET.SCALE_FACTOR = 0.0
cfg.DATASET.ROT_FACTOR = 0
cfg.DATASET.PROB_HALF_BODY = 0.0
cfg.DATASET.NUM_JOINTS_HALF_BODY = 0
cfg.DATASET.UPSAMPLE_FACTOR = 1
cfg.freeze()

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
val_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, True,
        transforms.Compose([
            transforms.ToTensor(),#把[0,255]形状为[H,W,C]的图片转化为[1,1.0]形状为[C,H,W]的torch.FloatTensor
            normalize,
        ])
    )

# 設定固定的圖像尺寸
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080

# 收集訓練集的邊界框數據
train_widths = []
train_heights = []
train_image_paths = []
train_right_edge_distances = []

# 收集驗證集的邊界框數據
val_widths = []
val_heights = []
val_image_paths = []
val_right_edge_distances = []

# 追蹤最大寬度和高度的記錄
max_width = 0
max_height = 0
max_width_image = ""
max_height_image = ""

# 從訓練資料集中讀取所有邊界框的寬度和高度
print(f"訓練集總共有 {len(train_dataset.db)} 個樣本")
for i, rec in enumerate(train_dataset.db):
    if i % 1000 == 0:
        print(f"處理訓練樣本 {i}/{len(train_dataset.db)}")
    
    # 檢查是否有w和h字段
    if 'w' in rec and 'h' in rec and 'center' in rec:
        w = rec['w']
        h = rec['h']
        center_x = rec['center'][0]
        image_path = rec.get('image', '')  # 獲取圖像路徑，如果不存在則為空字符串
        
        # 確保寬度和高度都大於0
        if w <= 0 or h <= 0:
            print(f"警告：訓練樣本 {i} 的寬度或高度不正確 (w={w}, h={h})，跳過此樣本")
            continue
        
        # 計算邊界框右邊緣到圖像右邊緣的距離
        box_right_edge = center_x + w/2
        right_distance = IMAGE_WIDTH - box_right_edge
        
        train_widths.append(w)
        train_heights.append(h)
        train_image_paths.append(image_path)
        train_right_edge_distances.append(right_distance)
        
        # 更新最大寬度和高度的記錄
        if w > max_width:
            max_width = w
            max_width_image = image_path
        
        if h > max_height:
            max_height = h
            max_height_image = image_path

# 從驗證資料集中讀取所有邊界框的寬度和高度
print(f"驗證集總共有 {len(val_dataset.db)} 個樣本")
for i, rec in enumerate(val_dataset.db):
    if i % 1000 == 0:
        print(f"處理驗證樣本 {i}/{len(val_dataset.db)}")
    
    # 檢查是否有w和h字段
    if 'w' in rec and 'h' in rec and 'center' in rec:
        w = rec['w']
        h = rec['h']
        center_x = rec['center'][0]
        image_path = rec.get('image', '')  # 獲取圖像路徑，如果不存在則為空字符串
        
        # 確保寬度和高度都大於0
        if w <= 0 or h <= 0:
            print(f"警告：驗證樣本 {i} 的寬度或高度不正確 (w={w}, h={h})，跳過此樣本")
            continue
        
        # 計算邊界框右邊緣到圖像右邊緣的距離
        box_right_edge = center_x + w/2
        right_distance = IMAGE_WIDTH - box_right_edge
        
        val_widths.append(w)
        val_heights.append(h)
        val_image_paths.append(image_path)
        val_right_edge_distances.append(right_distance)
        
        # 更新最大寬度和高度的記錄
        if w > max_width:
            max_width = w
            max_width_image = image_path
        
        if h > max_height:
            max_height = h
            max_height_image = image_path

# 檢查是否有收集到數據
if not train_widths and not val_widths:
    print("錯誤：沒有找到有效的邊界框數據。請檢查數據集格式。")
    sys.exit(1)

# 打印最大寬度和高度的圖像信息
print("\n最大寬度信息:")
print(f"最大寬度: {max_width} 像素")
print(f"最大寬度圖像路徑: {max_width_image}")

print("\n最大高度信息:")
print(f"最大高度: {max_height} 像素")
print(f"最大高度圖像路徑: {max_height_image}")

# 計算訓練集和驗證集的寬高比
train_aspect_ratios = [w/h for w, h in zip(train_widths, train_heights)]
val_aspect_ratios = [w/h for w, h in zip(val_widths, val_heights)]

# 找出訓練集的寬高比最大和最小的圖像
if train_aspect_ratios:
    train_max_aspect_ratio_idx = np.argmax(train_aspect_ratios)
    train_min_aspect_ratio_idx = np.argmin(train_aspect_ratios)

    print("\n訓練集寬高比信息:")
    print(f"最大寬高比: {train_aspect_ratios[train_max_aspect_ratio_idx]:.2f} (寬: {train_widths[train_max_aspect_ratio_idx]}, 高: {train_heights[train_max_aspect_ratio_idx]})")
    print(f"最大寬高比圖像路徑: {train_image_paths[train_max_aspect_ratio_idx]}")
    print(f"最小寬高比: {train_aspect_ratios[train_min_aspect_ratio_idx]:.2f} (寬: {train_widths[train_min_aspect_ratio_idx]}, 高: {train_heights[train_min_aspect_ratio_idx]})")
    print(f"最小寬高比圖像路徑: {train_image_paths[train_min_aspect_ratio_idx]}")

# 找出驗證集的寬高比最大和最小的圖像
if val_aspect_ratios:
    val_max_aspect_ratio_idx = np.argmax(val_aspect_ratios)
    val_min_aspect_ratio_idx = np.argmin(val_aspect_ratios)

    print("\n驗證集寬高比信息:")
    print(f"最大寬高比: {val_aspect_ratios[val_max_aspect_ratio_idx]:.2f} (寬: {val_widths[val_max_aspect_ratio_idx]}, 高: {val_heights[val_max_aspect_ratio_idx]})")
    print(f"最大寬高比圖像路徑: {val_image_paths[val_max_aspect_ratio_idx]}")
    print(f"最小寬高比: {val_aspect_ratios[val_min_aspect_ratio_idx]:.2f} (寬: {val_widths[val_min_aspect_ratio_idx]}, 高: {val_heights[val_min_aspect_ratio_idx]})")
    print(f"最小寬高比圖像路徑: {val_image_paths[val_min_aspect_ratio_idx]}")

# 繪製散點圖，同時顯示訓練集和驗證集的數據
plt.figure(figsize=(10, 8))
plt.scatter(train_widths, train_heights, alpha=0.5, s=5, color='blue', label='Train')
plt.scatter(val_widths, val_heights, alpha=0.5, s=5, color='red', label='Validation')
plt.title('Bounding Box Width-Height Distribution')
plt.xlabel('Width (pixels)')
plt.ylabel('Height (pixels)')
plt.grid(True, linestyle='--', alpha=0.7)

# 添加訓練集和驗證集的統計資訊
if train_widths:
    plt.axvline(x=np.mean(train_widths), color='blue', linestyle='--', 
                label=f'Train Mean Width: {np.mean(train_widths):.2f}')
    plt.axhline(y=np.mean(train_heights), color='blue', linestyle='-', 
                label=f'Train Mean Height: {np.mean(train_heights):.2f}')

if val_widths:
    plt.axvline(x=np.mean(val_widths), color='red', linestyle='--', 
                label=f'Val Mean Width: {np.mean(val_widths):.2f}')
    plt.axhline(y=np.mean(val_heights), color='red', linestyle='-', 
                label=f'Val Mean Height: {np.mean(val_heights):.2f}')

# 添加文本統計信息
y_pos = 0.95
if train_widths:
    plt.text(0.02, y_pos, f'Train Min Width: {np.min(train_widths):.2f}, Max Width: {np.max(train_widths):.2f}', 
             transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
    y_pos -= 0.05
    plt.text(0.02, y_pos, f'Train Min Height: {np.min(train_heights):.2f}, Max Height: {np.max(train_heights):.2f}', 
             transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
    y_pos -= 0.05

if val_widths:
    plt.text(0.02, y_pos, f'Val Min Width: {np.min(val_widths):.2f}, Max Width: {np.max(val_widths):.2f}', 
             transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
    y_pos -= 0.05
    plt.text(0.02, y_pos, f'Val Min Height: {np.min(val_heights):.2f}, Max Height: {np.max(val_heights):.2f}', 
             transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))

plt.legend()
plt.show()

# 額外繪製寬高比的直方圖，同時顯示訓練集和驗證集的數據
plt.figure(figsize=(10, 6))
if train_aspect_ratios:
    plt.hist(train_aspect_ratios, bins=50, alpha=0.7, color='blue', label='Train')
if val_aspect_ratios:
    plt.hist(val_aspect_ratios, bins=50, alpha=0.5, color='red', label='Validation')

plt.title('Bounding Box Aspect Ratio (Width/Height) Distribution')
plt.xlabel('Aspect Ratio (Width/Height)')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.7)

# 添加平均寬高比的垂直線
if train_aspect_ratios:
    plt.axvline(x=np.mean(train_aspect_ratios), color='blue', linestyle='--', 
                label=f'Train Mean Aspect Ratio: {np.mean(train_aspect_ratios):.2f}')
if val_aspect_ratios:
    plt.axvline(x=np.mean(val_aspect_ratios), color='red', linestyle='--', 
                label=f'Val Mean Aspect Ratio: {np.mean(val_aspect_ratios):.2f}')

plt.legend()
plt.show()

# 新增：繪製邊界框到圖像右邊緣的距離分佈，同時顯示訓練集和驗證集的數據
plt.figure(figsize=(12, 6))
if train_right_edge_distances:
    plt.hist(train_right_edge_distances, bins=50, alpha=0.7, color='blue', label='Train')
if val_right_edge_distances:
    plt.hist(val_right_edge_distances, bins=50, alpha=0.5, color='red', label='Validation')

plt.title('Distance from Bounding Box Right Edge to Image Right Edge')
plt.xlabel('Distance (pixels)')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.7)

# 添加統計信息
y_pos = 0.95
if train_right_edge_distances:
    train_mean_distance = np.mean(train_right_edge_distances)
    train_median_distance = np.median(train_right_edge_distances)
    # plt.axvline(x=train_mean_distance, color='blue', linestyle='--', 
    #             label=f'Train Mean Distance: {train_mean_distance:.2f}')
    # plt.axvline(x=train_median_distance, color='blue', linestyle=':', 
    #             label=f'Train Median Distance: {train_median_distance:.2f}')
    
    plt.text(0.02, y_pos, f'Train Min Distance: {np.min(train_right_edge_distances):.2f}, Max: {np.max(train_right_edge_distances):.2f}', 
             transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
    y_pos -= 0.05
    # plt.text(0.02, y_pos, f'Train Std Dev: {np.std(train_right_edge_distances):.2f}', 
    #          transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
    # y_pos -= 0.05
    
    # 計算訓練集中負值的百分比（表示邊界框超出圖像右邊緣）
    # train_negative_count = sum(1 for d in train_right_edge_distances if d < 0)
    # train_negative_percent = (train_negative_count / len(train_right_edge_distances)) * 100
    # plt.text(0.02, y_pos, f'Train boxes beyond right edge: {train_negative_count} ({train_negative_percent:.2f}%)', 
    #          transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
    # y_pos -= 0.05

if val_right_edge_distances:
    val_mean_distance = np.mean(val_right_edge_distances)
    val_median_distance = np.median(val_right_edge_distances)
    # plt.axvline(x=val_mean_distance, color='red', linestyle='--', 
    #             label=f'Val Mean Distance: {val_mean_distance:.2f}')
    # plt.axvline(x=val_median_distance, color='red', linestyle=':', 
    #             label=f'Val Median Distance: {val_median_distance:.2f}')
    
    plt.text(0.02, y_pos, f'Val Min Distance: {np.min(val_right_edge_distances):.2f}, Max: {np.max(val_right_edge_distances):.2f}', 
             transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
    y_pos -= 0.05
    # plt.text(0.02, y_pos, f'Val Std Dev: {np.std(val_right_edge_distances):.2f}', 
    #          transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
    # y_pos -= 0.05
    
    # 計算驗證集中負值的百分比（表示邊界框超出圖像右邊緣）
    # val_negative_count = sum(1 for d in val_right_edge_distances if d < 0)
    # val_negative_percent = (val_negative_count / len(val_right_edge_distances)) * 100
    # plt.text(0.02, y_pos, f'Val boxes beyond right edge: {val_negative_count} ({val_negative_percent:.2f}%)', 
    #          transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))

plt.legend()
plt.show()

print(f"處理完成！訓練集分析了 {len(train_widths)} 個有效樣本，驗證集分析了 {len(val_widths)} 個有效樣本。")