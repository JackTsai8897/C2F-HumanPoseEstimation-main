import numpy as np
import matplotlib.pyplot as plt
import random

# 設定隨機種子以確保結果可重現
seed = 42
np.random.seed(seed)
random.seed(seed)

# 設定要產生的點數量
num_points = 550

# 使用 random.uniform 產生 x 和 y 座標
x_coords = [random.uniform(0, 1) for _ in range(num_points)]
y_coords = [random.uniform(0, 1) for _ in range(num_points)]

# 創建圖表
plt.figure(figsize=(10, 8))
plt.scatter(x_coords, y_coords, color='blue', alpha=0.6, edgecolors='navy')

# 設定圖表屬性
plt.title('Random Point Distribution in range of x=[0, 1] and y=[0, 1]', fontsize=16)
plt.xlabel('X coord', fontsize=14)
plt.ylabel('Y coord', fontsize=14)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(True, linestyle='--', alpha=0.7)

# 添加邊框
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axhline(y=1, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=1, color='k', linestyle='-', alpha=0.3)

# 顯示圖表
plt.tight_layout()
plt.show()

# 輸出前10個點的座標，以便檢查
print("前10個隨機點的座標：")
for i in range(10):
    print(f"點 {i+1}: ({x_coords[i]:.4f}, {y_coords[i]:.4f})")


# 使用 random.uniform 產生 x, y 和 z 座標
x_coords = [random.uniform(0, 1) for _ in range(num_points)]
y_coords = [random.uniform(0, 1) for _ in range(num_points)]
z_coords = [random.uniform(0, 1) for _ in range(num_points)]

# 創建3D圖表
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# 繪製散點圖
scatter = ax.scatter(x_coords, y_coords, z_coords, 
                    c=z_coords,  # 使用z值作為顏色映射
                    cmap='viridis',  # 使用viridis顏色映射
                    s=50,  # 點的大小
                    alpha=0.8,  # 透明度
                    edgecolors='w')  # 白色邊框

# 設定圖表屬性
ax.set_title('Distribution in range of [0, 1] x [0, 1] x [0, 1]', fontsize=16)
ax.set_xlabel('X coord', fontsize=14)
ax.set_ylabel('Y coord', fontsize=14)
ax.set_zlabel('Z coord', fontsize=14)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)

# 添加顏色條
cbar = fig.colorbar(scatter, ax=ax, shrink=0.7)
cbar.set_label('Z value', fontsize=12)

# 添加網格
ax.grid(True, linestyle='--', alpha=0.3)

# 添加立方體邊框
# 底面
ax.plot([0, 1], [0, 0], [0, 0], 'k-', alpha=0.3)
ax.plot([0, 0], [0, 1], [0, 0], 'k-', alpha=0.3)
ax.plot([1, 1], [0, 1], [0, 0], 'k-', alpha=0.3)
ax.plot([0, 1], [1, 1], [0, 0], 'k-', alpha=0.3)
# 頂面
ax.plot([0, 1], [0, 0], [1, 1], 'k-', alpha=0.3)
ax.plot([0, 0], [0, 1], [1, 1], 'k-', alpha=0.3)
ax.plot([1, 1], [0, 1], [1, 1], 'k-', alpha=0.3)
ax.plot([0, 1], [1, 1], [1, 1], 'k-', alpha=0.3)
# 連接底面和頂面的線
ax.plot([0, 0], [0, 0], [0, 1], 'k-', alpha=0.3)
ax.plot([1, 1], [0, 0], [0, 1], 'k-', alpha=0.3)
ax.plot([0, 0], [1, 1], [0, 1], 'k-', alpha=0.3)
ax.plot([1, 1], [1, 1], [0, 1], 'k-', alpha=0.3)

# 設定視角
ax.view_init(elev=30, azim=45)

# 顯示圖表
plt.tight_layout()
plt.show()

# 輸出前10個點的座標，以便檢查
print("前10個隨機點的座標：")
for i in range(10):
    print(f"點 {i+1}: ({x_coords[i]:.4f}, {y_coords[i]:.4f}, {z_coords[i]:.4f})")