import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import random

# 使用與前面相同的數據
seed = 42
np.random.seed(seed)
random.seed(seed)

# 生成數據：6個維度，100個點
#abcdefghijklmnopqrstuvwxyz
num_points = 550
dimensions = ['x', 'y', 'z', 'w', 'v', 'u']

# 使用random.uniform生成每個維度的值，範圍都在[0, 1]
data = {}
for dim in dimensions:
    data[dim] = [random.uniform(0, 1) for _ in range(num_points)]

# 創建DataFrame
df = pd.DataFrame(data)

# 應用PCA降維到2D
pca_2d = PCA(n_components=2)
principalComponents_2d = pca_2d.fit_transform(df)
principal_df_2d = pd.DataFrame(data = principalComponents_2d, 
                             columns = ['Principal Component 1', 'Principal Component 2'])

# 應用PCA降維到3D
pca_3d = PCA(n_components=3)
principalComponents_3d = pca_3d.fit_transform(df)
principal_df_3d = pd.DataFrame(data = principalComponents_3d, 
                             columns = ['Principal Component 1', 'Principal Component 2', 'Principal Component 3'])

# 創建一個圖形，包含2D和3D PCA可視化
fig = plt.figure(figsize=(15, 6))

# 2D PCA散點圖
ax1 = fig.add_subplot(121)
ax1.scatter(principal_df_2d['Principal Component 1'], principal_df_2d['Principal Component 2'], 
           alpha=0.8, edgecolor='w')
ax1.set_xlabel('Principal Component 1', fontsize=12)
ax1.set_ylabel('Principal Component 2', fontsize=12)
ax1.set_title('PCA - 2D projection', fontsize=14)
ax1.grid(True, linestyle='--', alpha=0.7)

# 3D PCA散點圖
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(principal_df_3d['Principal Component 1'], 
           principal_df_3d['Principal Component 2'], 
           principal_df_3d['Principal Component 3'],
           alpha=0.8, edgecolor='w')
ax2.set_xlabel('Principal Component 1', fontsize=10)
ax2.set_ylabel('Principal Component 2', fontsize=10)
ax2.set_zlabel('Principal Component 3', fontsize=10)
ax2.set_title('PCA - 3D projection', fontsize=14)

plt.suptitle('PCA - Visualization of Dimensionality Reduction of {}-Dimensional Data'.format(len(dimensions)), fontsize=16, y=0.98)
plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.show()

# 顯示每個主成分的解釋方差比例
explained_variance_ratio_2d = pca_2d.explained_variance_ratio_
explained_variance_ratio_3d = pca_3d.explained_variance_ratio_

print("2D PCA解釋方差比例:")
for i, ratio in enumerate(explained_variance_ratio_2d):
    print(f"主成分 {i+1}: {ratio*100:.2f}%")
print(f"總解釋方差: {sum(explained_variance_ratio_2d)*100:.2f}%")

print("\n3D PCA解釋方差比例:")
for i, ratio in enumerate(explained_variance_ratio_3d):
    print(f"主成分 {i+1}: {ratio*100:.2f}%")
print(f"總解釋方差: {sum(explained_variance_ratio_3d)*100:.2f}%")


from sklearn.manifold import TSNE

# 生成數據：6個維度，100個點
#abcdefghijklmnopqrstuvwxyz
num_points = 550
dimensions = ['x', 'y', 'z', 'w', 'v', 'u']

# 使用random.uniform生成每個維度的值，範圍都在[0, 1]
data = {}
for dim in dimensions:
    data[dim] = [random.uniform(0, 1) for _ in range(num_points)]

# 創建DataFrame
df = pd.DataFrame(data)

# 應用t-SNE降維到2D
tsne_2d = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_results_2d = tsne_2d.fit_transform(df)
tsne_df_2d = pd.DataFrame(data=tsne_results_2d, 
                         columns=['t-SNE 1', 't-SNE 2'])

# 應用t-SNE降維到3D
tsne_3d = TSNE(n_components=3, random_state=42, perplexity=30)
tsne_results_3d = tsne_3d.fit_transform(df)
tsne_df_3d = pd.DataFrame(data=tsne_results_3d, 
                         columns=['t-SNE 1', 't-SNE 2', 't-SNE 3'])

# 創建一個圖形，包含2D和3D t-SNE可視化
fig = plt.figure(figsize=(15, 6))

# 2D t-SNE散點圖
ax1 = fig.add_subplot(121)
ax1.scatter(tsne_df_2d['t-SNE 1'], tsne_df_2d['t-SNE 2'], 
           alpha=0.8, edgecolor='w')
ax1.set_xlabel('t-SNE 1', fontsize=12)
ax1.set_ylabel('t-SNE 2', fontsize=12)
ax1.set_title('t-SNE - 2D projection', fontsize=14)
ax1.grid(True, linestyle='--', alpha=0.7)

# 3D t-SNE散點圖
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(tsne_df_3d['t-SNE 1'], 
           tsne_df_3d['t-SNE 2'], 
           tsne_df_3d['t-SNE 3'],
           alpha=0.8, edgecolor='w')
ax2.set_xlabel('t-SNE 1', fontsize=10)
ax2.set_ylabel('t-SNE 2', fontsize=10)
ax2.set_zlabel('t-SNE 3', fontsize=10)
ax2.set_title('t-SNE - 3D projection', fontsize=14)

plt.suptitle('t-SNE - Visualization of Dimensionality Reduction of {}-Dimensional Data'.format(len(dimensions)), fontsize=16, y=0.98)
plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.show()

import seaborn as sns

# 使用random.uniform生成每個維度的值，範圍都在[0, 1]
data = {}
for dim in dimensions:
    data[dim] = [random.uniform(0, 1) for _ in range(num_points)]

# 創建DataFrame
df = pd.DataFrame(data)

# 繪製散點矩陣圖
plt.figure(figsize=(12, 10))
sns.set_theme(style="ticks")
scatter_matrix = sns.pairplot(df, diag_kind="kde", markers="o", 
                             plot_kws=dict(s=50, edgecolor="white", linewidth=1),
                             diag_kws=dict(fill=True))

# 添加垂直線到對角線圖表（KDE圖）Kernel Density Estimation
# 遍歷對角線圖表（KDE圖）
for i, ax in enumerate(scatter_matrix.diag_axes):
    # 獲取當前KDE圖的y軸限制
    y_min, y_max = ax.get_ylim()
    
    # 添加垂直線在x=0, x=0.5, x=1的位置
    ax.vlines(x=0, ymin=y_min, ymax=y_max, colors='red', linestyles='--', linewidth=1.5)
    ax.vlines(x=0.5, ymin=y_min, ymax=y_max, colors='red', linestyles='--', linewidth=1.5)
    ax.vlines(x=1, ymin=y_min, ymax=y_max, colors='red', linestyles='--', linewidth=1.5)


scatter_matrix.figure.suptitle('Scatter Matrix - Visualization of {}-Dimensional Data'.format(len(dimensions)), y=1.02, fontsize=16)
plt.tight_layout()
plt.show()