import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import sys

import _init_paths
from dataset.HandKeypointsDataset import HandKeypointsDataset


def display_gaussian_noise_effect(image_path, std_values=None):
    """
    顯示應用不同標準差的高斯噪聲後的圖像效果
    
    Args:
        image_path: 輸入圖像的路徑
        std_values: 標準差值列表，如果為None，則使用默認值 [5, 15, 30, 50]
    """
    # 讀取圖像
    image = cv2.imread(image_path)
    if image is None:
        print(f"無法讀取圖像: {image_path}")
        return
    
    # 將圖像從BGR轉換為RGB (用於matplotlib顯示)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 如果沒有提供標準差值，則使用默認值
    if std_values is None:
        std_values = [5, 15, 30, 50]
    
    # 創建一個臨時的HandKeypointsDataset實例，僅用於調用add_gaussian_noise方法
    # 由於我們只需要使用add_gaussian_noise方法，所以可以創建一個空的實例
    dataset = HandKeypointsDataset.__new__(HandKeypointsDataset)
    
    # 設置圖像顯示
    fig, axes = plt.subplots(1, len(std_values) + 1, figsize=(15, 5))
    
    # 顯示原始圖像
    axes[0].imshow(image_rgb)
    axes[0].set_title('原始圖像')
    axes[0].axis('off')
    
    # 顯示不同標準差的噪聲效果
    for i, std in enumerate(std_values):
        # 應用高斯噪聲
        noisy_image = dataset.add_gaussian_noise(image, mean=0, std=std)
        
        # 轉換為RGB用於顯示
        noisy_image_rgb = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)
        
        # 顯示噪聲圖像
        axes[i + 1].imshow(noisy_image_rgb)
        axes[i + 1].set_title(f'Std = {std}')
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    # plt.savefig('gaussian_noise_comparison.png')
    plt.show()

def display_random_std_range(image_path, std_range=(0, 50), num_samples=4):
    """
    顯示在指定標準差範圍內隨機生成的高斯噪聲效果
    
    Args:
        image_path: 輸入圖像的路徑
        std_range: 標準差的範圍，默認為(0, 50)
        num_samples: 要生成的樣本數量，默認為4
    """
    # 讀取圖像
    image = cv2.imread(image_path)
    if image is None:
        print(f"無法讀取圖像: {image_path}")
        return
    
    # 將圖像從BGR轉換為RGB (用於matplotlib顯示)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 創建一個臨時的HandKeypointsDataset實例
    dataset = HandKeypointsDataset.__new__(HandKeypointsDataset)
    
    # 設置圖像顯示
    fig, axes = plt.subplots(1, num_samples + 1, figsize=(15, 5))
    
    # 顯示原始圖像
    axes[0].imshow(image_rgb)
    axes[0].set_title('原始圖像')
    axes[0].axis('off')
    
    # 顯示不同隨機標準差的噪聲效果
    for i in range(num_samples):
        # 在範圍內隨機生成標準差
        std = random.uniform(std_range[0], std_range[1])
        
        # 應用高斯噪聲
        noisy_image = dataset.add_gaussian_noise(image, mean=0, std=std)
        
        # 轉換為RGB用於顯示
        noisy_image_rgb = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)
        
        # 顯示噪聲圖像
        axes[i + 1].imshow(noisy_image_rgb)
        axes[i + 1].set_title(f'Rand Std = {std:.2f}')
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    #plt.savefig('random_std_gaussian_noise.png')
    plt.show()

if __name__ == "__main__":
    # 請替換為您自己的圖像路徑
    image_path = "../data/mydataset/images/train/0_eryi_img028.png"
    
    # 如果命令行提供了圖像路徑，則使用它
    # if len(sys.argv) > 1:
    #     image_path = sys.argv[1]
    
    print(f"使用圖像: {image_path}")
    
    # 顯示固定標準差的效果
    display_gaussian_noise_effect(image_path)
    
    # 顯示隨機標準差範圍的效果
    display_random_std_range(image_path, std_range=(0, 50))