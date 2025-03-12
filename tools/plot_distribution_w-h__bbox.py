import os.path as osp
import sys

import _init_paths
from config import cfg
from config import update_config

import dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

'''
# Update config from experiments
cfg.defrost()
cfg.merge_from_file("../experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml")
# cfg.merge_from_list("../experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml")
cfg.DATASET.ROOT = os.path.join(
        "..", cfg.DATA_DIR, cfg.DATASET.ROOT
    )
cfg.DATASET.SCALE_FACTOR = 0.0
cfg.DATASET.ROT_FACTOR = 0
cfg.DATASET.PROB_HALF_BODY = 0.0
cfg.DATASET.NUM_JOINTS_HALF_BODY = 0
cfg.freeze()
'''
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

# 收集所有邊界框的寬度和高度
widths = []
heights = []
image_paths = []  # 存儲圖像路徑

# 追蹤最大寬度和高度的記錄
max_width = 0
max_height = 0
max_width_image = ""
max_height_image = ""

# 從資料集中讀取所有邊界框的寬度和高度
print(f"總共有 {len(train_dataset.db)} 個樣本")
for i, rec in enumerate(train_dataset.db):
    if i % 1000 == 0:
        print(f"處理樣本 {i}/{len(train_dataset.db)}")
    
    # 檢查是否有w和h字段
    if 'w' in rec and 'h' in rec:
        w = rec['w']
        h = rec['h']
        image_path = rec.get('image', '')  # 獲取圖像路徑，如果不存在則為空字符串
        
        widths.append(w)
        heights.append(h)
        image_paths.append(image_path)
        
        # 更新最大寬度和高度的記錄
        if w > max_width:
            max_width = w
            max_width_image = image_path
        
        if h > max_height:
            max_height = h
            max_height_image = image_path

# 打印最大寬度和高度的圖像信息
print("\n最大寬度信息:")
print(f"最大寬度: {max_width} 像素")
print(f"最大寬度圖像路徑: {max_width_image}")

print("\n最大高度信息:")
print(f"最大高度: {max_height} 像素")
print(f"最大高度圖像路徑: {max_height_image}")

# 找出寬高比最大和最小的圖像
aspect_ratios = [w/h for w, h in zip(widths, heights)]
max_aspect_ratio_idx = np.argmax(aspect_ratios)
min_aspect_ratio_idx = np.argmin(aspect_ratios)

print("\n寬高比信息:")
print(f"最大寬高比: {aspect_ratios[max_aspect_ratio_idx]:.2f} (寬: {widths[max_aspect_ratio_idx]}, 高: {heights[max_aspect_ratio_idx]})")
print(f"最大寬高比圖像路徑: {image_paths[max_aspect_ratio_idx]}")

print(f"最小寬高比: {aspect_ratios[min_aspect_ratio_idx]:.2f} (寬: {widths[min_aspect_ratio_idx]}, 高: {heights[min_aspect_ratio_idx]})")
print(f"最小寬高比圖像路徑: {image_paths[min_aspect_ratio_idx]}")

# 繪製散點圖
plt.figure(figsize=(10, 8))
plt.scatter(widths, heights, alpha=0.5, s=5)
plt.title('Bounding Box Width-Height Distribution')
plt.xlabel('Width (pixels)')
plt.ylabel('Height (pixels)')
plt.grid(True, linestyle='--', alpha=0.7)

# 添加一些統計資訊
plt.axvline(x=np.mean(widths), color='r', linestyle='--', label=f'Mean Width: {np.mean(widths):.2f}')
plt.axhline(y=np.mean(heights), color='g', linestyle='--', label=f'Mean Height: {np.mean(heights):.2f}')

# 計算寬高比的分佈
aspect_ratios = [w/h for w, h in zip(widths, heights)]
# plt.text(0.02, 0.85, f'Mean Aspect Ratio (w/h): {np.mean(aspect_ratios):.2f}', 
#          transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
plt.text(0.02, 0.95, f'Min Width: {np.min(widths):.2f}, Max Width: {np.max(widths):.2f}', 
         transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
plt.text(0.02, 0.90, f'Min Height: {np.min(heights):.2f}, Max Height: {np.max(heights):.2f}', 
         transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))

plt.legend()

# 保存圖片
# output_dir = 'bbox_distribution_plots'
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
# plt.savefig(os.path.join(output_dir, 'width_height_distribution.png'), dpi=300)
# print(f"圖表已保存到 {os.path.join(output_dir, 'width_height_distribution.png')}")

# 顯示圖表
plt.show()

# 額外繪製寬高比的直方圖
plt.figure(figsize=(10, 6))
plt.hist(aspect_ratios, bins=50, alpha=0.7, color='blue')
plt.title('Bounding Box Aspect Ratio (Width/Height) Distribution')
plt.xlabel('Aspect Ratio (Width/Height)')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.7)
plt.axvline(x=np.mean(aspect_ratios), color='r', linestyle='--', 
            label=f'Mean Aspect Ratio: {np.mean(aspect_ratios):.2f}')
plt.legend()

# # 保存寬高比直方圖
# plt.savefig(os.path.join(output_dir, 'aspect_ratio_distribution.png'), dpi=300)
# print(f"寬高比直方圖已保存到 {os.path.join(output_dir, 'aspect_ratio_distribution.png')}")

# 顯示寬高比直方圖
plt.show()