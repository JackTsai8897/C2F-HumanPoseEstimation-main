import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import math
import copy

# 創建可視化目錄
os.makedirs('visualization/results', exist_ok=True)

# 讀取JSON文件
path = r'C:\Users\Jack\workspace\MyProject\C2F-HumanPoseEstimation-main\data\mydataset\annotations\right_hand_train.json'

# 創建一個模擬的JSON數據結構
data = {
    "images": [],
    "annotations": [],
    "categories": [
        {
            "id": 1,
            "name": "hand",
            "keypoints": ["E0", "E1", "E2", "W0", "W1", "W2"],
            "skeleton": []
        }
    ]
}

# 讀取JSON數據
try:
    with open(path, 'r') as f:
        data = json.load(f)
    print(f"成功讀取JSON文件: {path}")
except Exception as e:
    print(f"無法讀取JSON文件: {e}")
    # 使用提供的數據進行分析
    # 這裡可以直接使用您提供的annotation數據
    print("使用提供的數據進行分析")

# 創建映射: image_id -> file_name
image_id_to_filename = {}
for img in data.get("images", []):
    image_id_to_filename[img['id']] = img.get('file_name', f"unknown_{img['id']}")

# 獲取關鍵點名稱
keypoint_names = []
if data.get("categories") and len(data["categories"]) > 0:
    keypoint_names = data["categories"][0].get("keypoints", ["kp1", "kp2", "kp3", "kp4", "kp5", "kp6"])
num_keypoints = len(keypoint_names)

print(f"關鍵點名稱: {keypoint_names}")
print(f"關鍵點數量: {num_keypoints}")

# 定義對齊用的關鍵點索引
# 使用E0和W0作為對齊的關鍵點
elbow_idx = keypoint_names.index("E1") if "E1" in keypoint_names else 1
wrist_idx = keypoint_names.index("W1") if "W1" in keypoint_names else 4

print(f"對齊用的關鍵點: {keypoint_names[elbow_idx]} 和 {keypoint_names[wrist_idx]}")

# 創建字典來存儲每個subject的關鍵點數據，按圖像ID排序
subject_images = defaultdict(list)  # 存儲每個subject的image_id列表
subject_annotations = defaultdict(list)  # 存儲每個subject的annotations列表

# 創建字典來存儲每個subject的原始關鍵點數據
subject_keypoints = defaultdict(lambda: [[] for _ in range(num_keypoints)])

# 首先，收集每個subject的所有圖像和標註
for annotation in data.get("annotations", []):
    image_id = annotation['image_id']
    
    # 獲取文件名
    filename = image_id_to_filename.get(image_id, f"unknown_{image_id}")
    
    # 將文件名按"_img"分割，獲取subject名稱
    parts = filename.split('_img')
    subject_name = parts[0] if len(parts) > 1 else f"subject_{image_id//10}"
    
    # 存儲image_id和annotation
    subject_images[subject_name].append(image_id)
    subject_annotations[subject_name].append(annotation)
    
    # 獲取關鍵點坐標 (用於未對齊的分析)
    keypoints = annotation.get('keypoints', [])
    
    # 每3個值代表一個關鍵點的x, y, visibility
    for i in range(min(num_keypoints, len(keypoints) // 3)):
        x = keypoints[i*3]
        y = keypoints[i*3+1]
        visibility = keypoints[i*3+2]
        
        if visibility > 0:  # 只考慮可見的關鍵點
            subject_keypoints[subject_name][i].append((x, y))

# 計算每個subject每個關鍵點的均值和方差（使用歐氏距離）- 未對齊
results = {}
for subject, keypoints_list in subject_keypoints.items():
    results[subject] = {
        'mean': [],
        'variance': [],
        'std_dev': [],  # 標準差
        'count': []
    }
    
    for i, points in enumerate(keypoints_list):
        if points:  # 確保有數據點
            points_array = np.array(points)
            mean_x = np.mean(points_array[:, 0])
            mean_y = np.mean(points_array[:, 1])
            
            # 計算每個點到均值點的歐氏距離
            distances = [math.sqrt((p[0] - mean_x)**2 + (p[1] - mean_y)**2) for p in points]
            
            # 計算距離的方差和標準差
            distance_variance = np.var(distances)
            distance_std_dev = np.std(distances)
            
            results[subject]['mean'].append((mean_x, mean_y))
            results[subject]['variance'].append(distance_variance)
            results[subject]['std_dev'].append(distance_std_dev)
            results[subject]['count'].append(len(points))
        else:
            results[subject]['mean'].append((0, 0))
            results[subject]['variance'].append(0)
            results[subject]['std_dev'].append(0)
            results[subject]['count'].append(0)

# 計算每個subject的整體方差 - 未對齊
overall_variances = {}
for subject, result in results.items():
    overall_variance = np.mean(result['variance'])
    overall_std_dev = np.mean(result['std_dev'])
    overall_variances[subject] = overall_variance
    
    print(f"主體 {subject} (未對齊):")
    print(f"  平均距離方差: {overall_variance:.2f}")
    print(f"  平均距離標準差: {overall_std_dev:.2f}")

# 為每個subject繪製原始關鍵點分布圖
for subject, result in results.items():
    plt.figure(figsize=(12, 10))
    plt.title(f'Subject: {subject} - Original Keypoint Distribution')
    
    # 繪製每個關鍵點的位置
    for i in range(num_keypoints):
        if result['count'][i] > 0:
            points = subject_keypoints[subject][i]
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            
            # 繪製散點圖
            plt.scatter(x_coords, y_coords, label=f"{keypoint_names[i]} (n={result['count'][i]}, σ={result['std_dev'][i]:.2f})")
            
            # 繪製均值點 (較大)
            mean_x, mean_y = result['mean'][i]
            plt.scatter(mean_x, mean_y, s=100, marker='X', color='red')
            
            # 繪製標準差圓圈
            std_dev = result['std_dev'][i]
            circle = plt.Circle((mean_x, mean_y), std_dev, fill=False, color='red', linestyle='--', alpha=0.5)
            plt.gca().add_patch(circle)
            
            # 添加標籤
            plt.annotate(keypoint_names[i] if i < len(keypoint_names) else f"keypoint_{i}", 
                        (mean_x, mean_y), textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    # Y軸反轉，因為圖像坐標系的原點在左上角
    plt.gca().invert_yaxis()
    plt.savefig(f'visualized_results/mydataset/aligned_results/{subject}_original_keypoint_distribution.png')
    plt.close()

# 為每個subject獲取參考關鍵點（第一張圖像）
reference_points = {}
for subject, annotations in subject_annotations.items():
    # 獲取第一個annotation
    if annotations:
        first_anno = annotations[0]
        keypoints = first_anno.get('keypoints', [])
        
        # 確保有足夠的關鍵點
        if len(keypoints) >= max(elbow_idx, wrist_idx) * 3 + 3:
            # 獲取參考關鍵點
            elbow_x = keypoints[elbow_idx*3]
            elbow_y = keypoints[elbow_idx*3+1]
            wrist_x = keypoints[wrist_idx*3]
            wrist_y = keypoints[wrist_idx*3+1]
            
            reference_points[subject] = {
                'elbow': (elbow_x, elbow_y),
                'wrist': (wrist_x, wrist_y),
                'angle': math.atan2(elbow_y - wrist_y, elbow_x - wrist_x),
                'scale': math.sqrt((wrist_x - elbow_x)**2 + (wrist_y - elbow_y)**2)
            }
            print(f"Subject {subject} 參考點設置完成")
        else:
            print(f"Subject {subject} 的第一個標註缺少足夠的關鍵點")
    else:
        print(f"Subject {subject} 沒有標註數據")

# 創建字典來存儲對齊後的關鍵點數據
aligned_subject_keypoints = defaultdict(lambda: [[] for _ in range(num_keypoints)])

# 對齊並收集關鍵點數據
for subject, annotations in subject_annotations.items():
    if subject not in reference_points:
        print(f"跳過Subject {subject}，因為沒有參考點")
        continue
    
    ref = reference_points[subject]
    ref_elbow = ref['elbow']
    ref_wrist = ref['wrist']
    ref_angle = ref['angle']
    ref_scale = ref['scale']
    
    for annotation in annotations:
        keypoints = annotation.get('keypoints', [])
        src_elbow_x = keypoints[elbow_idx*3]
        src_elbow_y = keypoints[elbow_idx*3+1]
        src_wrist_x = keypoints[wrist_idx*3]
        src_wrist_y = keypoints[wrist_idx*3+1]
        
        src_elbow = (src_elbow_x, src_elbow_y)
        src_wrist = (src_wrist_x, src_wrist_y)
        src_scale = math.sqrt((src_wrist_x - src_elbow_x)**2 + (src_wrist_y - src_elbow_y)**2)
        src_angle = math.atan2(src_elbow_y - src_wrist_y, src_elbow_x - src_wrist_x)
        angle_diff = ref_angle - src_angle
        
        sf = src_scale/ref_scale
        
        # 每3個值代表一個關鍵點的x, y, visibility
        for i in range(min(num_keypoints, len(keypoints) // 3)):
            x = keypoints[i*3]
            y = keypoints[i*3+1]
            visibility = keypoints[i*3+2]
            
            if visibility > 0:  # 只考慮可見的關鍵點
                # 1. 相對於當前手腕的偏移（使手腕位於原點）
                dx = x - src_wrist_x
                dy = y - src_wrist_y

                # 2. 旋轉校正
                rotated_x = dx * math.cos(angle_diff) - dy * math.sin(angle_diff)
                rotated_y = dx * math.sin(angle_diff) + dy * math.cos(angle_diff)

                # 3. 平移到參考手腕位置
                aligned_x = rotated_x*sf + ref_wrist[0]
                aligned_y = rotated_y*sf + ref_wrist[1]
                
                # 存儲對齊後的坐標
                aligned_subject_keypoints[subject][i].append((aligned_x, aligned_y))

# 計算對齊後的每個subject每個關鍵點的均值和方差
aligned_results = {}
for subject, keypoints_list in aligned_subject_keypoints.items():
    aligned_results[subject] = {
        'mean': [],
        'variance': [],
        'std_dev': [],  # 標準差
        'count': []
    }
    
    for i, points in enumerate(keypoints_list):
        if points:  # 確保有數據點
            points_array = np.array(points)
            mean_x = np.mean(points_array[:, 0])
            mean_y = np.mean(points_array[:, 1])
            
            # 計算每個點到均值點的歐氏距離
            distances = [math.sqrt((p[0] - mean_x)**2 + (p[1] - mean_y)**2) for p in points]
            
            # 計算距離的方差和標準差
            distance_variance = np.var(distances)
            distance_std_dev = np.std(distances)
            
            aligned_results[subject]['mean'].append((mean_x, mean_y))
            aligned_results[subject]['variance'].append(distance_variance)
            aligned_results[subject]['std_dev'].append(distance_std_dev)
            aligned_results[subject]['count'].append(len(points))
        else:
            aligned_results[subject]['mean'].append((0, 0))
            aligned_results[subject]['variance'].append(0)
            aligned_results[subject]['std_dev'].append(0)
            aligned_results[subject]['count'].append(0)

# 輸出對齊後的結果
print("\n對齊後的分析結果:")
for subject, result in aligned_results.items():
    print(f"\n主體: {subject}")
    print(f"數據點數量: {sum(result['count'])}")
    
    for i in range(num_keypoints):
        kp_name = keypoint_names[i] if i < len(keypoint_names) else f"keypoint_{i}"
        mean = result['mean'][i]
        variance = result['variance'][i]
        std_dev = result['std_dev'][i]
        count = result['count'][i]
        
        print(f"  關鍵點 {kp_name}:")
        print(f"    數量: {count}")
        print(f"    均值位置: X={mean[0]:.2f}, Y={mean[1]:.2f}")
        print(f"    距離方差: {variance:.2f}")
        print(f"    距離標準差: {std_dev:.2f}")

# 為每個subject繪製對齊後的關鍵點分布圖
for subject, result in aligned_results.items():
    plt.figure(figsize=(12, 10))
    plt.title(f'Subject: {subject} - Aligned Keypoint Distribution')
    
    # 繪製每個關鍵點的位置
    for i in range(num_keypoints):
        if result['count'][i] > 0:
            points = aligned_subject_keypoints[subject][i]
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            
            # 繪製散點圖
            plt.scatter(x_coords, y_coords, label=f"{keypoint_names[i]} (n={result['count'][i]}, σ={result['std_dev'][i]:.2f})")
            
            # 繪製均值點 (較大)
            mean_x, mean_y = result['mean'][i]
            plt.scatter(mean_x, mean_y, s=100, marker='X', color='red')
            
            # 繪製標準差圓圈
            std_dev = result['std_dev'][i]
            circle = plt.Circle((mean_x, mean_y), std_dev, fill=False, color='red', linestyle='--', alpha=0.5)
            plt.gca().add_patch(circle)
            
            # 添加標籤
            plt.annotate(keypoint_names[i] if i < len(keypoint_names) else f"keypoint_{i}", 
                        (mean_x, mean_y), textcoords="offset points", xytext=(0,10), ha='center')
    
    # 繪製參考線 (原點是手腕位置，X軸指向index_mcp)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.plot([0, 1], [0, 0], 'k-', alpha=0.5)  # 從手腕到標準化的index_mcp位置
    
    plt.xlabel('Normalized X Coordinate')
    plt.ylabel('Normalized Y Coordinate')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'visualized_results/mydataset/aligned_results/{subject}_aligned_keypoint_distribution.png')
    plt.close()

# 創建所有subject的對齊關鍵點疊加圖
plt.figure(figsize=(15, 12))
plt.title('All Subjects - Aligned Keypoint Distribution')

# 為每個subject使用不同的顏色
colors = plt.cm.tab10(np.linspace(0, 1, len(aligned_results)))
subject_colors = {subject: colors[i] for i, subject in enumerate(aligned_results.keys())}

# 繪製每個subject的均值點
for subject, result in aligned_results.items():
    for i in range(num_keypoints):
        if result['count'][i] > 0:
            mean_x, mean_y = result['mean'][i]
            plt.scatter(mean_x, mean_y, s=80, marker='o', color=subject_colors[subject], alpha=0.7)
            plt.annotate(f"{subject}-{keypoint_names[i]}", 
                        (mean_x, mean_y), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)

# 繪製參考線
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.plot([0, 1], [0, 0], 'k-', alpha=0.5)  # 從手腕到標準化的index_mcp位置

plt.xlabel('Normalized X Coordinate')
plt.ylabel('Normalized Y Coordinate')
plt.grid(True)
plt.savefig('visualized_results/mydataset/aligned_results/all_subjects_aligned_keypoints.png')
plt.close()

# 繪製對齊後的方差比較圖
plt.figure(figsize=(15, 10))
plt.title('Aligned Keypoint Distance Variance Comparison Across Subjects')

bar_width = 0.8 / len(aligned_results)
index = np.arange(num_keypoints)

for i, (subject, result) in enumerate(aligned_results.items()):
    variances = result['variance']
    
    # 只顯示前8個subject，避免圖表過於擁擠
    if i < 8:
        plt.bar(index + i*bar_width, variances, bar_width, alpha=0.7, label=f'{subject}')

plt.xlabel('Keypoint')
plt.ylabel('Distance Variance (Aligned)')
plt.xticks(index + bar_width*len(aligned_results)/2, [kp if i < len(keypoint_names) else f"kp{i}" for i, kp in enumerate(keypoint_names)])
plt.legend()
plt.tight_layout()
plt.savefig('visualized_results/mydataset/aligned_results/aligned_keypoint_variance_comparison.png')
plt.close()

# 計算每個subject的對齊後整體方差
print("\n每個subject的對齊後整體穩定性:")
aligned_overall_variances = {}
aligned_overall_std_dev = {}
for subject, result in aligned_results.items():
    overall_variance = np.mean(result['variance'])
    overall_std_dev = np.mean(result['std_dev'])
    aligned_overall_variances[subject] = overall_variance
    aligned_overall_std_dev[subject] = overall_std_dev
    
    print(f"主體 {subject}:")
    print(f"  平均距離方差: {overall_variance:.2f}")
    print(f"  平均距離標準差: {overall_std_dev:.2f}")

# 繪製每個subject的對齊後整體方差比較圖
subjects = list(aligned_results.keys())
std_dev_values = [aligned_overall_std_dev[subject] for subject in subjects]

plt.figure(figsize=(12, 8))
plt.title('Overall Distance Variance Comparison Between Subjects (Aligned)')
plt.bar(subjects, std_dev_values, alpha=0.7)
plt.xlabel('Subject')
plt.ylabel('Average Distance Variance (Aligned)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('visualized_results/mydataset/aligned_results/aligned_overall_variance_comparison.png')
plt.close()

# 繪製熱力圖，顯示每個subject每個關鍵點的對齊後方差
import matplotlib.cm as cm

# 創建一個包含所有subject的熱力圖
plt.figure(figsize=(14, 10))
plt.title('Aligned Keypoint Distance Variance Heatmap Across All Subjects')

# 準備熱力圖數據
heatmap_data = np.zeros((len(aligned_results), num_keypoints))
for i, subject in enumerate(aligned_results.keys()):
    for j in range(num_keypoints):
        heatmap_data[i, j] = aligned_results[subject]['variance'][j]

plt.imshow(heatmap_data, cmap=cm.hot, aspect='auto')
plt.colorbar(label='Distance Variance (Aligned)')
plt.yticks(range(len(aligned_results)), list(aligned_results.keys()))
plt.xticks(range(num_keypoints), [kp if i < len(keypoint_names) else f"kp{i}" for i, kp in enumerate(keypoint_names)])
plt.tight_layout()
plt.savefig('visualized_results/mydataset/aligned_results/all_subjects_aligned_variance_heatmap.png')
plt.close()

# 找出對齊後最穩定和最不穩定的關鍵點
aligned_keypoint_avg_variance = [0] * num_keypoints
for subject, result in aligned_results.items():
    for i in range(num_keypoints):
        aligned_keypoint_avg_variance[i] += result['variance'][i] / len(aligned_results)

aligned_most_stable_kp = keypoint_names[np.argmin(aligned_keypoint_avg_variance)]
aligned_least_stable_kp = keypoint_names[np.argmax(aligned_keypoint_avg_variance)]

print(f"\n對齊後最穩定的關鍵點: {aligned_most_stable_kp} (平均方差: {min(aligned_keypoint_avg_variance):.2f})")
print(f"對齊後最不穩定的關鍵點: {aligned_least_stable_kp} (平均方差: {max(aligned_keypoint_avg_variance):.2f})")

# 找出對齊後最穩定和最不穩定的subject
aligned_most_stable_subject = min(aligned_overall_variances.items(), key=lambda x: x[1])[0]
aligned_least_stable_subject = max(aligned_overall_variances.items(), key=lambda x: x[1])[0]

print(f"\n對齊後最穩定的subject: {aligned_most_stable_subject} (平均方差: {aligned_overall_variances[aligned_most_stable_subject]:.2f})")
print(f"對齊後最不穩定的subject: {aligned_least_stable_subject} (平均方差: {aligned_overall_variances[aligned_least_stable_subject]:.2f})")

# 比較對齊前後的方差變化
print("\n對齊前後方差變化:")
for subject in subjects:
    if subject in overall_variances and subject in aligned_overall_variances:
        before = overall_variances[subject]
        after = aligned_overall_variances[subject]
        change = (after - before) / before * 100 if before > 0 else float('inf')
        
        print(f"主體 {subject}:")
        print(f"  對齊前方差: {before:.2f}")
        print(f"  對齊後方差: {after:.2f}")
        print(f"  變化百分比: {change:.2f}%")

# 繪製對齊前後的方差比較圖
plt.figure(figsize=(14, 8))
plt.title('Variance Comparison Before and After Alignment')

x = np.arange(len(subjects))
width = 0.35

plt.bar(x - width/2, [overall_variances.get(subject, 0) for subject in subjects], width, label='Before Alignment')
plt.bar(x + width/2, [aligned_overall_variances.get(subject, 0) for subject in subjects], width, label='After Alignment')

plt.xlabel('Subject')
plt.ylabel('Average Distance Variance')
plt.xticks(x, subjects, rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('visualized_results/mydataset/aligned_results/before_after_alignment_comparison.png')
plt.close()

print("\n對齊分析完成！所有圖表已保存到 visualized_results/mydataset/aligned_results/ 目錄")