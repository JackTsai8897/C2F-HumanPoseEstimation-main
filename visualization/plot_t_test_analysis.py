# Configuration
orientation = 'left'
JSON_FILE1 = r'C:/Users/Jack/workspace/MyProject/C2F-HumanPoseEstimation-main/output/mydataset/pose_hrnet/new_w48_512x224_adam_lr1e-3/per_image_results_{}.json'.format(orientation) 
JSON_FILE2 = r'C:/Users/Jack/workspace/MyProject/C2F-HumanPoseEstimation-main/output/mydataset/pose_hrnet/{}_w48_512x224_adam_lr1e-3/per_image_results.json'.format(orientation)
OUTPUT_DIR = "visualized_results/mydataset/analysis/t_test"

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib.ticker as ticker

# 創建輸出目錄
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 讀取JSON文件
def load_json_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# 提取subject名稱
def extract_subject_name(image_name):
    return image_name.split('_img')[0] if '_img' in image_name else 'unknown'

# 載入數據
data1 = load_json_data(JSON_FILE1)
data2 = load_json_data(JSON_FILE2)

# 準備數據框架 - 根據JSON格式調整
def prepare_dataframe(data, model_name):
    rows = []
    for item in data:
        image_name = item['image_name']
        subject_name = extract_subject_name(image_name)
        
        # 處理每個關鍵點
        for kp in item['keypoints']:
            keypoint_id = kp['keypoint_id']
            keypoint_name = kp['keypoint_name']
            distance = kp['distance']  # 這是誤差值
            
            rows.append({
                'subject': subject_name,
                'keypoint': keypoint_name,
                'keypoint_id': keypoint_id,
                'error': distance,
                'model': model_name,
                'image_name': image_name
            })
    
    return pd.DataFrame(rows)

# 創建數據框架
df1 = prepare_dataframe(data1, 'model1')
df2 = prepare_dataframe(data2, 'model2')
combined_df = pd.concat([df1, df2])

# 獲取所有subject和keypoint
all_subjects = sorted(combined_df['subject'].unique())
all_keypoints = sorted(combined_df['keypoint'].unique())

# 進行t檢驗分析
def perform_t_test(group1, group2):
    if len(group1) > 1 and len(group2) > 1:  # 確保有足夠的樣本
        t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
        return t_stat, p_value
    else:
        return np.nan, np.nan

# 創建結果表格 - 使用純數值DataFrame
results = pd.DataFrame(index=all_subjects + ['All_s'], columns=all_keypoints + ['All_kp'])
results = results.astype('float64')  # 確保所有值都是浮點數

# 進行每個subject和keypoint的t檢驗
for subject in all_subjects + ['All_s']:
    for keypoint in all_keypoints + ['All_kp']:
        if subject == 'All_s' and keypoint == 'All_kp':
            # 所有subject和所有keypoint
            group1 = df1['error'].values
            group2 = df2['error'].values
        elif subject == 'All_s':
            # 所有subject，特定keypoint
            group1 = df1[df1['keypoint'] == keypoint]['error'].values
            group2 = df2[df2['keypoint'] == keypoint]['error'].values
        elif keypoint == 'All_kp':
            # 特定subject，所有keypoint
            group1 = df1[df1['subject'] == subject]['error'].values
            group2 = df2[df2['subject'] == subject]['error'].values
        else:
            # 特定subject和特定keypoint
            group1 = df1[(df1['subject'] == subject) & (df1['keypoint'] == keypoint)]['error'].values
            group2 = df2[(df2['subject'] == subject) & (df2['keypoint'] == keypoint)]['error'].values
        
        if len(group1) > 0 and len(group2) > 0:
            _, p_value = perform_t_test(group1, group2)
            results.loc[subject, keypoint] = p_value if not np.isnan(p_value) else np.nan
        else:
            results.loc[subject, keypoint] = np.nan

# 可視化p值熱圖
plt.figure(figsize=(16, 12))
# 創建掩碼以隱藏NaN值
mask = results.isna()

# 檢查數據是否包含任何有效值
if not results.isna().all().all():
    # 使用簡單的熱圖方法，避免複雜的數據處理
    ax = sns.heatmap(
        results,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu_r",
        mask=mask,
        vmin=0,
        vmax=0.05,
        cbar_kws={'label': 'p-value'},
        annot_kws={"size": 24}  # 調整數值字體大小為8
    )
    plt.title('T-test P-values (Model1 vs Model2)', fontsize=24)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'p_values_heatmap.png'), dpi=300)
else:
    print("No valid p-values to plot in heatmap")
plt.close()

# 計算平均誤差差異
mean_diff = {}
for subject in all_subjects + ['All_s']:
    for keypoint in all_keypoints + ['All_kp']:
        if subject == 'All_s' and keypoint == 'All_kp':
            mean1 = df1['error'].mean()
            mean2 = df2['error'].mean()
        elif subject == 'All_s':
            mean1 = df1[df1['keypoint'] == keypoint]['error'].mean()
            mean2 = df2[df2['keypoint'] == keypoint]['error'].mean()
        elif keypoint == 'All_kp':
            mean1 = df1[df1['subject'] == subject]['error'].mean()
            mean2 = df2[df2['subject'] == subject]['error'].mean()
        else:
            mean1 = df1[(df1['subject'] == subject) & (df1['keypoint'] == keypoint)]['error'].mean()
            mean2 = df2[(df2['subject'] == subject) & (df2['keypoint'] == keypoint)]['error'].mean()
        
        key = f"{subject}_{keypoint}"
        mean_diff[key] = mean1 - mean2

# 創建顯著性差異摘要
significant_diff = results < 0.05
significant_count = significant_diff.sum().sum()
total_tests = (~results.isna()).sum().sum()

# 生成摘要報告
summary = f"""
T-test Analysis Summary:
-----------------------
Total tests conducted: {total_tests}
Significant differences found (p < 0.05): {significant_count} ({significant_count/total_tests*100:.2f}% if total_tests > 0 else 0)

Model comparison:
- Model 1: {os.path.basename(JSON_FILE1)}
- Model 2: {os.path.basename(JSON_FILE2)}
"""

# 找出最顯著的差異
significant_pairs = []
for subject in all_subjects + ['All_s']:
    for keypoint in all_keypoints + ['All_kp']:
        p_value = results.loc[subject, keypoint]
        if pd.notnull(p_value) and p_value < 0.05:
            key = f"{subject}_{keypoint}"
            diff = mean_diff.get(key, 0)
            better_model = "Model 1" if diff < 0 else "Model 2"
            significant_pairs.append((subject, keypoint, p_value, abs(diff), better_model))

significant_pairs.sort(key=lambda x: x[2])  # 按p值排序

# 將顯著差異添加到摘要
if significant_pairs:
    summary += "\nMost significant differences (p < 0.05):\n"
    summary += "Subject, Keypoint, P-value, Abs Diff, Better Model\n"
    for subject, keypoint, p_value, diff, better_model in significant_pairs[:10]:  # 顯示前10個
        summary += f"{subject}, {keypoint}, {p_value:.5f}, {diff:.5f}, {better_model}\n"

# 保存摘要報告
with open(os.path.join(OUTPUT_DIR, 'analysis_summary.txt'), 'w') as f:
    f.write(summary)

# 為每個subject創建箱形圖比較
for subject in all_subjects + ['All_s']:
    try:
        plt.figure(figsize=(16, 10))
        
        if subject == 'All_s':
            subject_df = combined_df
            title = "All Subjects - Error Comparison"
        else:
            subject_df = combined_df[combined_df['subject'] == subject]
            title = f"Subject: {subject} - Error Comparison"
        
        # 確保有數據可繪製
        if subject_df.empty:
            print(f"No data for subject {subject}, skipping boxplot")
            plt.close()
            continue
            
        # 確保關鍵點按ID排序
        keypoints_in_subject = subject_df['keypoint'].unique()
        if len(keypoints_in_subject) > 0:
            order = sorted(keypoints_in_subject)
            
            ax = sns.boxplot(x='keypoint', y='error', hue='model', data=subject_df, order=order)
            plt.title(title, fontsize=16)
            plt.xlabel('Keypoint', fontsize=12)
            plt.ylabel('Error (Distance)', fontsize=12)
            plt.xticks(rotation=45)
            
            # 添加顯著性標記
            if subject != 'All_s':
                for i, kp in enumerate(order):
                    if kp in results.columns and pd.notnull(results.loc[subject, kp]) and results.loc[subject, kp] < 0.05:
                        max_error = subject_df[subject_df['keypoint'] == kp]['error'].max()
                        plt.text(i, max_error + 0.5, '*', ha='center', va='center', color='red', fontsize=20)
            
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f'boxplot_{subject}.png'), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error creating boxplot for {subject}: {e}")
        plt.close()

# 創建整體比較圖
try:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='model', y='error', data=combined_df)
    plt.title('Overall Error Comparison', fontsize=16)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Error (Distance)', fontsize=12)

    # 添加均值線和標籤
    for i, model in enumerate(['model1', 'model2']):
        mean_val = combined_df[combined_df['model'] == model]['error'].mean()
        plt.axhline(y=mean_val, color='r', linestyle='--', alpha=0.3)
        plt.text(i, mean_val + 0.2, f'Mean: {mean_val:.2f}', ha='center')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'overall_comparison.png'), dpi=300)
except Exception as e:
    print(f"Error creating overall comparison: {e}")
plt.close()

# 創建每個關鍵點的比較圖
try:
    plt.figure(figsize=(12, 8))
    keypoint_means = combined_df.groupby(['keypoint', 'model'])['error'].mean().unstack()
    if not keypoint_means.empty:
        keypoint_means.plot(kind='bar', yerr=combined_df.groupby(['keypoint', 'model'])['error'].std().unstack())
        plt.title('Mean Error by Keypoint', fontsize=16)
        plt.xlabel('Keypoint', fontsize=12)
        plt.ylabel('Mean Error', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'keypoint_comparison.png'), dpi=300)
except Exception as e:
    print(f"Error creating keypoint comparison chart: {e}")
plt.close()

# 創建熱圖顯示平均誤差差異
try:
    # 創建一個純數值的DataFrame，包含所有subjects和keypoints以及All_s和All_kp
    mean_diff_df = pd.DataFrame(index=all_subjects + ['All_s'], columns=all_keypoints + ['All_kp'])
    mean_diff_df = mean_diff_df.astype('float64')  # 確保所有值都是浮點數
    
    # 填充所有組合的平均誤差差異
    for subject in all_subjects + ['All_s']:
        for keypoint in all_keypoints + ['All_kp']:
            key = f"{subject}_{keypoint}"
            if key in mean_diff:
                mean_diff_df.loc[subject, keypoint] = mean_diff[key]

    plt.figure(figsize=(16, 12))
    # 創建掩碼以隱藏NaN值
    mask = mean_diff_df.isna()
    
    if not mean_diff_df.isna().all().all():
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(
            mean_diff_df,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            mask=mask,
            center=0,
            cbar_kws={'label': 'Mean Error Difference (Model1 - Model2)'},
            annot_kws={"size": 24}  # 調整數值字體大小為8
        )
        plt.title('Mean Error Difference by Subject and Keypoint', fontsize=24)
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'mean_diff_heatmap.png'), dpi=300)
    else:
        print("No valid data for mean difference heatmap")
except Exception as e:
    print(f"Error creating mean difference heatmap: {e}")
plt.close()

print(f"Analysis complete. Results saved to {OUTPUT_DIR}")