import json
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def visualize_annotations(json_file, image_dir, output_dir=None, show_images=True, save_images=True):
    """
    根據JSON檔案在圖像上繪製邊界框和關鍵點
    
    參數:
        json_file: JSON檔案路徑
        image_dir: 圖像目錄路徑
        output_dir: 輸出目錄路徑，如果不指定則在image_dir下創建'visualized'子目錄
        show_images: 是否顯示圖像
        save_images: 是否保存圖像
    """
    # 讀取JSON檔案
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # 創建輸出目錄
    if save_images:
        if output_dir is None:
            output_dir = os.path.join(image_dir, 'visualized')
        os.makedirs(output_dir, exist_ok=True)
    
    # 獲取類別信息，用於顯示名稱
    categories = {cat['id']: cat['name'] for cat in data['categories']}
    
    # 獲取關鍵點名稱
    keypoint_names = {}
    for cat in data['categories']:
        keypoint_names[cat['id']] = cat['keypoints']
    
    # 創建圖像ID到文件名的映射
    image_id_to_filename = {}
    image_id_to_size = {}
    
    # 檢查JSON數據中是否包含images字段
    if 'images' in data:
        for img in data['images']:
            image_id_to_filename[img['id']] = img['file_name']
            if 'width' in img and 'height' in img:
                image_id_to_size[img['id']] = (img['width'], img['height'])
    else:
        # 如果沒有images字段，則使用默認的文件名格式
        print("警告: JSON數據中沒有找到images字段，將使用默認的文件名格式")
        for ann in data['annotations']:
            image_id = ann['image_id']
            image_id_to_filename[image_id] = f"{image_id:06d}.jpg"
    

    # 按照圖像ID對標註進行分組
    annotations_by_image = {}
    for ann in data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)
    
    # 處理每張圖像
    for image_id, annotations in annotations_by_image.items():
        # 獲取圖像文件名
        if image_id in image_id_to_filename:
            image_filename = image_id_to_filename[image_id]
        else:
            print(f"找不到圖像ID {image_id} 的文件名，跳過...")
            continue
        
        # 構建圖像路徑
        image_path = os.path.join(image_dir, image_filename)
        
        # 檢查圖像是否存在
        if not os.path.exists(image_path):
            print(f"無法找到圖像 {image_path}，跳過...")
            continue
        
        # 讀取圖像
        image = cv2.imread(image_path)
        if image is None:
            print(f"無法讀取圖像 {image_path}，跳過...")
            continue
        
        # 將BGR轉換為RGB用於matplotlib顯示
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 為每個標註繪製邊界框和關鍵點
        for ann in annotations:
            # 獲取邊界框
            bbox = ann['bbox']  # [x, y, width, height]
            x, y, w, h = [int(coord) for coord in bbox]
            
            # 獲取類別名稱
            category_id = ann['category_id']
            category_name = categories.get(category_id, f"類別 {category_id}")
            
            # 繪製邊界框
            cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image_rgb, category_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # 獲取關鍵點
            keypoints = ann['keypoints']
            num_keypoints = ann['num_keypoints']
            
            # 關鍵點格式: [x1, y1, v1, x2, y2, v2, ...]
            # v: 0=不可見, 1=可見但被遮擋, 2=可見
            keypoint_colors = [
                (255, 0, 0),    # W0: 藍色
                (0, 255, 0),    # W1: 綠色
                (0, 0, 255),    # W2: 紅色
                (255, 255, 0),  # E0: 青色
                (255, 0, 255),  # E1: 洋紅色
                (0, 255, 255)   # E2: 黃色
            ]
            
            # 繪製關鍵點
            for i in range(num_keypoints):
                kp_x = keypoints[i*3]
                kp_y = keypoints[i*3 + 1]
                kp_v = keypoints[i*3 + 2]
                
                if kp_v > 0:  # 如果關鍵點可見
                    # 繪製關鍵點
                    cv2.circle(image_rgb, (int(kp_x), int(kp_y)), 5, keypoint_colors[i], -1)
                    
                    # 添加關鍵點名稱
                    kp_name = keypoint_names.get(category_id, [])[i] if i < len(keypoint_names.get(category_id, [])) else f"kp{i}"
                    cv2.putText(image_rgb, kp_name, (int(kp_x) + 5, int(kp_y) - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, keypoint_colors[i], 1)
        
        # 顯示圖像
        if show_images:
            plt.figure(figsize=(12, 8))
            plt.imshow(image_rgb)
            plt.title(f"圖像 ID: {image_id}")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        
        # 保存圖像
        if save_images:
            output_path = os.path.join(output_dir, f"visualized_{image_filename}")
            # 轉回BGR格式用於OpenCV保存
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, image_bgr)
            print(f"已保存可視化結果到: {output_path}")

def main():
    # 設置JSON檔案和圖像目錄路徑
    root = '../data/mydataset'
    dataset = 'val'
    json_file = '{}/annotations/person_keypoints_{}.json'.format(root, dataset)  # 請替換為您的JSON檔案路徑
    image_dir = '{}/images/{}'.format(root, dataset)            # 請替換為您的圖像目錄路徑
    output_dir = 'visualized_results/mydataset'  # 輸出目錄
    
    # 可視化標註
    visualize_annotations(json_file, image_dir, output_dir, show_images=True, save_images=True)

if __name__ == "__main__":
    main()