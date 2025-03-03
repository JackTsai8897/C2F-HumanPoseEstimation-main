# --- Start of .\import torch.py ---
import torch
print(torch.cuda.is_available())

print(torch.__version__)
# --- End of .\import torch.py ---

# --- Start of .\merge_script.py ---
import os

output_file = "merged_script.py"

def get_all_python_files(directory):
    """ 遞迴獲取目錄下所有 .py 檔案的完整路徑 """
    py_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                py_files.append(os.path.join(root, file))
    return py_files

# 取得所有 Python 檔案
python_files = get_all_python_files(".")

# 開啟輸出檔案，並將所有檔案內容寫入
with open(output_file, "w", encoding="utf-8") as outfile:
    for file_path in python_files:
        with open(file_path, encoding="utf-8") as infile:
            outfile.write(f"# --- Start of {file_path} ---\n")
            outfile.write(infile.read())
            outfile.write(f"\n# --- End of {file_path} ---\n\n")

print(f"所有 Python 檔案已合併到 {output_file}")
# --- End of .\merge_script.py ---

# --- Start of .\plot.py ---
import numpy as np
import matplotlib.pyplot as plt
import random
import torch

c = [0.0168, 0.1970, 0.1064, 0.0775, 0.0828, 0.0154, 0.0412, 0.1178, 0.0859,
       0.1793, 0.1327, 0.1043, 0.1377, 0.2248, 0.0880, 0.0869, 0.0484,0.0189, 0.2643, 0.0962, 0.0665, 0.1055, 0.0201, 0.0733, 0.1569, 0.0832,
        0.2178, 0.1882, 0.1841, 0.1570, 0.3463, 0.1118, 0.1254, 0.0609,0.0172, 0.1409, 0.1298, 0.0598, 0.0923, 0.0179, 0.0425, 0.1375, 0.1142,
        0.1659, 0.1278, 0.1320, 0.1909, 0.2120, 0.0994, 0.0803, 0.0374,0.0153, 0.1835, 0.0690, 0.0688, 0.0799, 0.0173, 0.0527, 0.1034, 0.0755,
        0.1760, 0.1203, 0.1074, 0.0734, 0.2114, 0.1041, 0.0858, 0.0428,0.0198, 0.2429, 0.1114, 0.0750, 0.1121, 0.0196, 0.0724, 0.1614, 0.1121,
        0.2140, 0.1872, 0.1919, 0.1742, 0.3346, 0.1093, 0.1258, 0.0594,0.0085, 0.0044, 0.0469, 0.0217, 0.0180, 0.0119, 0.0059, 0.0049, 0.0248,
        0.0099, 0.0060, 0.0095, 0.0257, 0.0233, 0.0632, 0.0058, 0.0288,0.0117, 0.0744, 0.0938, 0.0432, 0.0623, 0.0175, 0.0272, 0.0669, 0.0655,
        0.0984, 0.0533, 0.0725, 0.1232, 0.1204, 0.0883, 0.0533, 0.0230,0.0085, 0.0486, 0.0558, 0.0396, 0.0397, 0.0135, 0.0155, 0.0320, 0.0276,
        0.0628, 0.0308, 0.0392, 0.0415, 0.0736, 0.0735, 0.0263, 0.0201,0.0124, 0.1326, 0.0464, 0.0619, 0.0445, 0.0128, 0.0286, 0.0437, 0.0794,
        0.0924, 0.0791, 0.0436, 0.0319, 0.1091, 0.0789, 0.0501, 0.0371,0.0177, 0.2505, 0.0901, 0.0666, 0.0846, 0.0190, 0.0697, 0.1368, 0.0983,
        0.2222, 0.1712, 0.1673, 0.1470, 0.3295, 0.1095, 0.1200, 0.0536,0.0163, 0.1651, 0.1317, 0.0750, 0.0956, 0.0180, 0.0491, 0.1125, 0.1201,
        0.1701, 0.1353, 0.1088, 0.1715, 0.1963, 0.0932, 0.0911, 0.0454,0.0123, 0.1429, 0.0765, 0.0531, 0.0506, 0.0142, 0.0297, 0.0462, 0.0580,
        0.1201, 0.0732, 0.0608, 0.0677, 0.1367, 0.0864, 0.0517, 0.0490,0.0134, 0.1556, 0.0839, 0.0699, 0.0651, 0.0142, 0.0412, 0.0717, 0.0793,
        0.1356, 0.0987, 0.0697, 0.0565, 0.1479, 0.0776, 0.0663, 0.0431,0.0176, 0.1754, 0.0879, 0.0541, 0.0632, 0.0170, 0.0571, 0.1051, 0.0720,
        0.2029, 0.1188, 0.1432, 0.1380, 0.2802, 0.1000, 0.0951, 0.0362,0.0176, 0.1754, 0.0879, 0.0541, 0.0632, 0.0170, 0.0571, 0.1051, 0.0720,
        0.2029, 0.1188, 0.1432, 0.1380, 0.2802, 0.1000, 0.0951, 0.0362,0.0134, 0.1171, 0.0797, 0.0531, 0.0556, 0.0149, 0.0377, 0.0704, 0.0651,
        0.1322, 0.0797, 0.0842, 0.0985, 0.1482, 0.0964, 0.0667, 0.0302,0.0138, 0.0403, 0.1049, 0.0416, 0.0651, 0.0166, 0.0244, 0.0874, 0.0810,
        0.0927, 0.0793, 0.0833, 0.1603, 0.1036, 0.0983, 0.0380, 0.0226,0.0191, 0.2035, 0.1218, 0.0723, 0.1088, 0.0194, 0.0625, 0.1540, 0.1135,
        0.1883, 0.1562, 0.1564, 0.1901, 0.2611, 0.1048, 0.1094, 0.0515,0.0200, 0.2043, 0.1367, 0.0839, 0.1136, 0.0193, 0.0610, 0.1479, 0.1390,
        0.1912, 0.1652, 0.1324, 0.1942, 0.2423, 0.1049, 0.1066, 0.0567,0.0177, 0.1366, 0.1390, 0.0576, 0.0909, 0.0179, 0.0409, 0.1251, 0.0993,
        0.1593, 0.1303, 0.1335, 0.1794, 0.1946, 0.0984, 0.0749, 0.0351,0.0138, 0.1410, 0.0865, 0.0577, 0.0646, 0.0160, 0.0413, 0.0814, 0.0788,
        0.1395, 0.0992, 0.0829, 0.0948, 0.1407, 0.0921, 0.0681, 0.0337,0.0124, 0.1274, 0.0544, 0.0632, 0.0378, 0.0124, 0.0229, 0.0394, 0.0749,
        0.0794, 0.0716, 0.0402, 0.0272, 0.0933, 0.0706, 0.0439, 0.0351,0.0169, 0.1736, 0.0812, 0.0720, 0.0673, 0.0163, 0.0428, 0.0839, 0.0936,
        0.1613, 0.1072, 0.0770, 0.0852, 0.1729, 0.0968, 0.0772, 0.0483,0.0202, 0.2249, 0.1061, 0.0719, 0.0965, 0.0181, 0.0641, 0.1683, 0.1050,
        0.2186, 0.1827, 0.1601, 0.1611, 0.2846, 0.0968, 0.1191, 0.0596,0.0156, 0.1378, 0.1026, 0.0556, 0.0763, 0.0170, 0.0446, 0.1039, 0.0852,
        0.1665, 0.1068, 0.1029, 0.1054, 0.1714, 0.0901, 0.0758, 0.0361,0.0106, 0.0389, 0.0565, 0.0329, 0.0459, 0.0135, 0.0189, 0.0530, 0.0169,
        0.0840, 0.0476, 0.0556, 0.0542, 0.0829, 0.0839, 0.0252, 0.0198,0.0135, 0.1333, 0.0510, 0.0580, 0.0487, 0.0150, 0.0312, 0.0489, 0.0856,
        0.1060, 0.0835, 0.0481, 0.0461, 0.1201, 0.0880, 0.0533, 0.0435,0.0097, 0.0836, 0.0570, 0.0455, 0.0314, 0.0144, 0.0220, 0.0316, 0.0769,
        0.0628, 0.0502, 0.0353, 0.0335, 0.0650, 0.0955, 0.0367, 0.0309,0.0092, 0.0632, 0.0595, 0.0345, 0.0434, 0.0137, 0.0246, 0.0406, 0.0170,
        0.0950, 0.0453, 0.0675, 0.0443, 0.1056, 0.0863, 0.0262, 0.0250,0.0174, 0.2286, 0.0991, 0.0532, 0.0789, 0.0179, 0.0726, 0.1303, 0.0840,
        0.2243, 0.1409, 0.1890, 0.1665, 0.3649, 0.1072, 0.1122, 0.0451]
c = c *100
c = sorted(c)
c = torch.tensor(c)

s_qian=[]
for i in range(27000):
    s_qian.append(0)
# print(len(s_qian),s_qian)
s_hou = c[26999:-1]
# print(len(s_hou))
s = s_qian + s_hou.numpy().tolist()
s1 = s_qian + (s_hou*0.8).numpy().tolist()
average_loss = torch.mean(c,dim=[0])
# print(average_loss)
yf=1/(1+torch.exp(40*(-c+0.1)))
# print(yf)
d = yf * c
d_qian = d[0:35000]*0.8
d_hou = d[34999:-1]*0.5
d1=d_qian.numpy().tolist()+d_hou.numpy().tolist()
# plt.figure(1,figsize=(16,6))
print(len(c),len(s),len(d))
plt.plot(c, color="blue", linewidth=3.0, linestyle="--", label="MSE", alpha=0.9)  # 颜色 线宽 类型 标签 透明度
# plt.plot(c*0.9, color="blue", linewidth=3.0, linestyle="-", label="MSE", alpha=0.9)

plt.plot(s, "green", linewidth=3.0, linestyle="--",label="OHKM")  # r*:red *  颜色为红色，线型为*
# plt.plot(s1, "green", linewidth=3.0, linestyle="--",label="OHKM")

plt.plot(d, "red", linewidth=3.0, linestyle="--",label="Ours")
# plt.plot(d1, "red", linewidth=3.0, linestyle="--",label="Ours")
front1 = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 30,}
front2 = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 30,}
front3 = {'family' : 'SimHei',
        'weight' : 'normal',
        'size'   : 30,}

# plt.title("损失值统计图",front3)  # 标题
plt.xlabel("关键点索引",front3)
plt.ylabel('损失值',front3)

# 坐标轴的操作
# 坐标轴的位置
ax = plt.gca()  # 引入坐标轴
ax.spines["right"].set_color("none")
ax.spines["top"].set_color("none")
ax.spines["left"].set_position(("data", 0))
ax.spines["bottom"].set_position(("data", 0))

# 坐标轴的刻度显示位置
ax.xaxis.set_ticks_position("bottom")
ax.yaxis.set_ticks_position("left")

# 设置坐标的显示范围
plt.yticks(np.linspace(0, 0.4, 5, endpoint=True))  # 设置坐标的显示范围
# 设置刻度数字大小和边框
for lable in ax.get_xticklabels() + ax.get_yticklabels():
    lable.set_fontsize(25)  # 刻度大小
    lable.set_bbox(dict(facecolor="white", edgecolor="None", alpha=0.2))  # 刻度下面的小边框

# 图例
plt.legend(loc=(0.07,0.8),prop=front2)

# 网格线
# plt.grid()
foo_fig = plt.gcf()
foo_fig.savefig("cm.png",dpi = 600,bbox_inches="tight")
plt.show()

# def ssm_r_top():
#     ticks = [0.01, 0.05, 0.1, 0.15, 0.5, 1, 5, 10, 50, 100]
#     x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#     # x = [0.01, 0.05, 0.1, 0.15, 0.5,1, 5,10,50,100]
#     y = [80.62,81.45,83.55,81.90,82.31,82.03,79.52,78.69,73.59,70.59]
#     # plt.plot(x, y, label='SSM-r-top1')
#     # plt.plot(x, y, '.b')
#     plt.plot(c, color="blue", linewidth=3.0, linestyle="--", label="MSE", alpha=0.9)
#     plt.plot(s, "green", linewidth=3.0, linestyle="dashed", label="OHKM")
#     plt.plot(d1, "red", linewidth=3.0, linestyle="-", label="Ours")
#     # plt.xticks(x, ticks)
#     plt.xlabel("keypoints index", front2)
#     plt.ylabel('Loss value', front2)
#     plt.xlim([0,51000])
#     plt.ylim([0, 0.4])
#     plt.legend()
#
#     # plt调用gcf函数取得当前绘制的figure并调用savefig函数
#     foo_fig = plt.gcf()  # 'get current figure'
#     foo_fig.savefig('ssm_r_top.eps', format='eps', dpi=600)
#
#     plt.show()
#
# ssm_r_top()

# import numpy as np
# import matplotlib.pyplot as plt
# import random
# import torch
#
# c = [0.0168, 0.1970, 0.1064, 0.0775, 0.0828, 0.0154, 0.0412, 0.1178, 0.0859,
#        0.1793, 0.1327, 0.1043, 0.1377, 0.2248, 0.0880, 0.0869, 0.0484,0.0189, 0.2643, 0.0962, 0.0665, 0.1055, 0.0201, 0.0733, 0.1569, 0.0832,
#         0.2178, 0.1882, 0.1841, 0.1570, 0.3463, 0.1118, 0.1254, 0.0609,0.0172, 0.1409, 0.1298, 0.0598, 0.0923, 0.0179, 0.0425, 0.1375, 0.1142,
#         0.1659, 0.1278, 0.1320, 0.1909, 0.2120, 0.0994, 0.0803, 0.0374,0.0153, 0.1835, 0.0690, 0.0688, 0.0799, 0.0173, 0.0527, 0.1034, 0.0755,
#         0.1760, 0.1203, 0.1074, 0.0734, 0.2114, 0.1041, 0.0858, 0.0428,0.0198, 0.2429, 0.1114, 0.0750, 0.1121, 0.0196, 0.0724, 0.1614, 0.1121,
#         0.2140, 0.1872, 0.1919, 0.1742, 0.3346, 0.1093, 0.1258, 0.0594,0.0085, 0.0044, 0.0469, 0.0217, 0.0180, 0.0119, 0.0059, 0.0049, 0.0248,
#         0.0099, 0.0060, 0.0095, 0.0257, 0.0233, 0.0632, 0.0058, 0.0288,0.0117, 0.0744, 0.0938, 0.0432, 0.0623, 0.0175, 0.0272, 0.0669, 0.0655,
#         0.0984, 0.0533, 0.0725, 0.1232, 0.1204, 0.0883, 0.0533, 0.0230,0.0085, 0.0486, 0.0558, 0.0396, 0.0397, 0.0135, 0.0155, 0.0320, 0.0276,
#         0.0628, 0.0308, 0.0392, 0.0415, 0.0736, 0.0735, 0.0263, 0.0201,0.0124, 0.1326, 0.0464, 0.0619, 0.0445, 0.0128, 0.0286, 0.0437, 0.0794,
#         0.0924, 0.0791, 0.0436, 0.0319, 0.1091, 0.0789, 0.0501, 0.0371,0.0177, 0.2505, 0.0901, 0.0666, 0.0846, 0.0190, 0.0697, 0.1368, 0.0983,
#         0.2222, 0.1712, 0.1673, 0.1470, 0.3295, 0.1095, 0.1200, 0.0536,0.0163, 0.1651, 0.1317, 0.0750, 0.0956, 0.0180, 0.0491, 0.1125, 0.1201,
#         0.1701, 0.1353, 0.1088, 0.1715, 0.1963, 0.0932, 0.0911, 0.0454,0.0123, 0.1429, 0.0765, 0.0531, 0.0506, 0.0142, 0.0297, 0.0462, 0.0580,
#         0.1201, 0.0732, 0.0608, 0.0677, 0.1367, 0.0864, 0.0517, 0.0490,0.0134, 0.1556, 0.0839, 0.0699, 0.0651, 0.0142, 0.0412, 0.0717, 0.0793,
#         0.1356, 0.0987, 0.0697, 0.0565, 0.1479, 0.0776, 0.0663, 0.0431,0.0176, 0.1754, 0.0879, 0.0541, 0.0632, 0.0170, 0.0571, 0.1051, 0.0720,
#         0.2029, 0.1188, 0.1432, 0.1380, 0.2802, 0.1000, 0.0951, 0.0362,0.0176, 0.1754, 0.0879, 0.0541, 0.0632, 0.0170, 0.0571, 0.1051, 0.0720,
#         0.2029, 0.1188, 0.1432, 0.1380, 0.2802, 0.1000, 0.0951, 0.0362,0.0134, 0.1171, 0.0797, 0.0531, 0.0556, 0.0149, 0.0377, 0.0704, 0.0651,
#         0.1322, 0.0797, 0.0842, 0.0985, 0.1482, 0.0964, 0.0667, 0.0302,0.0138, 0.0403, 0.1049, 0.0416, 0.0651, 0.0166, 0.0244, 0.0874, 0.0810,
#         0.0927, 0.0793, 0.0833, 0.1603, 0.1036, 0.0983, 0.0380, 0.0226,0.0191, 0.2035, 0.1218, 0.0723, 0.1088, 0.0194, 0.0625, 0.1540, 0.1135,
#         0.1883, 0.1562, 0.1564, 0.1901, 0.2611, 0.1048, 0.1094, 0.0515,0.0200, 0.2043, 0.1367, 0.0839, 0.1136, 0.0193, 0.0610, 0.1479, 0.1390,
#         0.1912, 0.1652, 0.1324, 0.1942, 0.2423, 0.1049, 0.1066, 0.0567,0.0177, 0.1366, 0.1390, 0.0576, 0.0909, 0.0179, 0.0409, 0.1251, 0.0993,
#         0.1593, 0.1303, 0.1335, 0.1794, 0.1946, 0.0984, 0.0749, 0.0351,0.0138, 0.1410, 0.0865, 0.0577, 0.0646, 0.0160, 0.0413, 0.0814, 0.0788,
#         0.1395, 0.0992, 0.0829, 0.0948, 0.1407, 0.0921, 0.0681, 0.0337,0.0124, 0.1274, 0.0544, 0.0632, 0.0378, 0.0124, 0.0229, 0.0394, 0.0749,
#         0.0794, 0.0716, 0.0402, 0.0272, 0.0933, 0.0706, 0.0439, 0.0351,0.0169, 0.1736, 0.0812, 0.0720, 0.0673, 0.0163, 0.0428, 0.0839, 0.0936,
#         0.1613, 0.1072, 0.0770, 0.0852, 0.1729, 0.0968, 0.0772, 0.0483,0.0202, 0.2249, 0.1061, 0.0719, 0.0965, 0.0181, 0.0641, 0.1683, 0.1050,
#         0.2186, 0.1827, 0.1601, 0.1611, 0.2846, 0.0968, 0.1191, 0.0596,0.0156, 0.1378, 0.1026, 0.0556, 0.0763, 0.0170, 0.0446, 0.1039, 0.0852,
#         0.1665, 0.1068, 0.1029, 0.1054, 0.1714, 0.0901, 0.0758, 0.0361,0.0106, 0.0389, 0.0565, 0.0329, 0.0459, 0.0135, 0.0189, 0.0530, 0.0169,
#         0.0840, 0.0476, 0.0556, 0.0542, 0.0829, 0.0839, 0.0252, 0.0198,0.0135, 0.1333, 0.0510, 0.0580, 0.0487, 0.0150, 0.0312, 0.0489, 0.0856,
#         0.1060, 0.0835, 0.0481, 0.0461, 0.1201, 0.0880, 0.0533, 0.0435,0.0097, 0.0836, 0.0570, 0.0455, 0.0314, 0.0144, 0.0220, 0.0316, 0.0769,
#         0.0628, 0.0502, 0.0353, 0.0335, 0.0650, 0.0955, 0.0367, 0.0309,0.0092, 0.0632, 0.0595, 0.0345, 0.0434, 0.0137, 0.0246, 0.0406, 0.0170,
#         0.0950, 0.0453, 0.0675, 0.0443, 0.1056, 0.0863, 0.0262, 0.0250,0.0174, 0.2286, 0.0991, 0.0532, 0.0789, 0.0179, 0.0726, 0.1303, 0.0840,
#         0.2243, 0.1409, 0.1890, 0.1665, 0.3649, 0.1072, 0.1122, 0.0451]
# c = c *50
# c = sorted(c)
# c = torch.tensor(c)
#
#
#
# yf1=(np.exp(1*c)-1)/(np.exp(1*c)+1)
# d1 = yf1 * c
#
# yf2=(np.exp(15*c)-1)/(np.exp(15*c)+1)
# d2 = yf2 * c
#
# yf3=(np.exp(30*c)-1)/(np.exp(30*c)+1)
# d3 = yf3 * c
#
#
#
# plt.figure(1,figsize=(16,6))
# # print(len(c))
# plt.plot(c, color="blue", linewidth=3.0, linestyle="-", label="MSE", alpha=0.9)  # 颜色 线宽 类型 标签 透明度
# plt.plot(d1, color="red", linewidth=3.0, linestyle="-", label="γ=1", alpha=0.9)
# plt.plot(d2, "green", linewidth=3.0, linestyle="-",label="γ=20")  # r*:red *  颜色为红色，线型为*
# plt.plot(d3, "purple", linewidth=3.0, linestyle="-",label="γ=30")

front1 = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 30,}
front2 = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 16,}
front3 = {'family' : 'SimHei',
        'weight' : 'normal',
        'size'   : 16,}

# plt.title("损失值统计图",front3)  # 标题
# plt.xlabel("关键点索引",front3)
# plt.ylabel('损失值',front3)

# 坐标轴的操作
# 坐标轴的位置
# ax = plt.gca()  # 引入坐标轴
# ax.spines["right"].set_color("none")
# ax.spines["top"].set_color("none")
# ax.spines["left"].set_position(("data", 0))
# ax.spines["bottom"].set_position(("data", 0))

# 坐标轴的刻度显示位置
# ax.xaxis.set_ticks_position("bottom")
# ax.yaxis.set_ticks_position("left")
# import matplotlib
# matplotlib.rcParams['xtick.direction'] = 'in'
# matplotlib.rcParams['ytick.direction'] = 'in'
# 设置坐标的显示范围
# plt.yticks(np.linspace(0, 0.4, 5, endpoint=True))  # 设置坐标的显示范围
# 设置刻度数字大小和边框
# for lable in ax.get_xticklabels() + ax.get_yticklabels():
#     lable.set_fontsize(10)  # 刻度大小
#     lable.set_bbox(dict(facecolor="white", edgecolor="None", alpha=0.2))  # 刻度下面的小边框

# 图例
# plt.legend(loc=(0.07,0.6),prop=front2)

# 网格线
# plt.grid()
# plt.savefig("cm.png",dpi = 600,bbox_inches="tight")
# plt.show()

# def ssm_r_top():
#     # ticks = [0.01, 0.05, 0.1, 0.15, 0.5, 1, 5, 10, 50, 100]
#     # x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#     # x = [0.01, 0.05, 0.1, 0.15, 0.5,1, 5,10,50,100]
#     # y = [80.62,81.45,83.55,81.90,82.31,82.03,79.52,78.69,73.59,70.59]
#     # plt.plot(x, y, label='SSM-r-top1')
#     # plt.plot(x, y, '.b')
#     plt.plot(c, color="blue", linewidth=1.0, linestyle="-", label="MSE", alpha=0.9)  # 颜色 线宽 类型 标签 透明度
#     plt.plot(d1, color="red", linewidth=1.0, linestyle="-", label="γ=1", alpha=0.9)
#     plt.plot(d2, "green", linewidth=1.0, linestyle="-", label="γ=20")  # r*:red *  颜色为红色，线型为*
#     plt.plot(d3, "purple", linewidth=1.0, linestyle="-", label="γ=30")
#
#     plt.xlabel("Keypoints index", front2)
#     plt.ylabel('Loss value', front2)
#     plt.legend(loc=(0.07, 0.8), prop=front2)
#     plt.xlim([0, 25500])
#     plt.ylim([0, 0.4])
#     ax = plt.gca()
#     for lable in ax.get_xticklabels() + ax.get_yticklabels():
#         lable.set_fontsize(8)  # 刻度大小
#         lable.set_bbox(dict(facecolor="white", edgecolor="None", alpha=0.2))
#
#     # plt调用gcf函数取得当前绘制的figure并调用savefig函数
#     foo_fig = plt.gcf()  # 'get current figure'
#     foo_fig.savefig('ssm_r_top.jpg', format='jpg', dpi=600)
#
#     plt.show()
#
# ssm_r_top()
#
# def ssm_r_top1():
#     ticks = [0.01, 0.05, 0.1, 0.15, 0.5, 1, 5, 10, 50, 100]
#     x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#     # x = [0.01, 0.05, 0.1, 0.15, 0.5,1, 5,10,50,100]
#     y = [80.62,81.45,83.55,81.90,82.31,82.03,79.52,78.69,73.59,70.59]
#     plt.plot(x, y, label='SSM-r-top1')
#     plt.plot(x, y, '.b')
#     plt.xlabel("Keypoints index", front2)
#     plt.ylabel('Loss value', front2)
#     plt.legend(loc=(0.07, 0.8), prop=front2)
#     plt.xlim([0, 25])
#     plt.ylim([0, 0.4])
#     ax = plt.gca()
#     for lable in ax.get_xticklabels() + ax.get_yticklabels():
#         lable.set_fontsize(8)  # 刻度大小
#         lable.set_bbox(dict(facecolor="white", edgecolor="None", alpha=0.2))
#
#     # plt调用gcf函数取得当前绘制的figure并调用savefig函数
#     foo_fig = plt.gcf()  # 'get current figure'
#     foo_fig.savefig('ssm_r_top.jpg', format='jpg', dpi=600)
#
#     plt.show()
#
# ssm_r_top1()

# import numpy as np
# import matplotlib.pyplot as plt
# import random
#
# c = []
#
# plt.figure(1,figsize=(16,6))
#
# plt.plot(c, color="blue", linewidth=3.0, linestyle="--", label="MSE", alpha=0.9)  # 颜色 线宽 类型 标签 透明度
# # plt.plot(s, "green", linewidth=3.0, linestyle="dashed",label="OHKM")  # r*:red *  颜色为红色，线型为*
# # plt.plot(d, "red", linewidth=3.0, linestyle="dashed",label="Ours")
# front1 = {'family' : 'Times New Roman',
#         'weight' : 'normal',
#         'size'   : 30,}
# front2 = {'family' : 'Times New Roman',
#         'weight' : 'normal',
#         'size'   : 25,}
# plt.title("Loss statistics",front1)  # 标题
# plt.xlabel("Index of keypoints",front1)
# plt.ylabel('Loss',front1)
#
# # 坐标轴的操作
# # 坐标轴的位置
# ax = plt.gca()  # 引入坐标轴
# ax.spines["right"].set_color("none")
# ax.spines["top"].set_color("none")
# ax.spines["left"].set_position(("data", 0))
# ax.spines["bottom"].set_position(("data", 0))
#
# # 坐标轴的刻度显示位置
# ax.xaxis.set_ticks_position("bottom")
# ax.yaxis.set_ticks_position("left")
#
# # 设置坐标的显示范围
# plt.yticks(np.linspace(0, 0.4, 5, endpoint=True))  # 设置坐标的显示范围
# # 设置刻度数字大小和边框
# for lable in ax.get_xticklabels() + ax.get_yticklabels():
#     lable.set_fontsize(25)  # 刻度大小
#     lable.set_bbox(dict(facecolor="white", edgecolor="None", alpha=0.2))  # 刻度下面的小边框
#
# # 图例
# plt.legend(loc=(0.07,0.8),prop=front2)
#
# # 网格线
# # plt.grid()
# plt.savefig("cm.png",dpi = 300,bbox_inches="tight")
# plt.show()

# import matplotlib.pyplot as plt
# import numpy as np
#
#
# # x = [0, 0.5, 1, 1.5, 2.0, 2.5, 3, 3.5, 4, 4.5, 5]
# # y = [0,0.45,0.55,0.90,0.31,0.03,0.52,0.69,0.59,0.59]
# x = np.arange(0, 0.6, 0.05)
# # y = 2.0 / (1 + np.power(np.e, -2 * x)) - 1
# y = ((np.exp(20*x)-1) / (np.exp(20*x)+1))
#
# import matplotlib
# matplotlib.rcParams['xtick.direction'] = 'in'
# matplotlib.rcParams['ytick.direction'] = 'in'
# '''
# plt.plot(Abscissa, ordinate,"bo-",linewidth="",markersize="",label="")
# "bo-":The shape of the point of the curve
# linewidth:The distance from the point on the curve
# markersize:The size of the point on the curve
# label:Description of the curve
# '''
# plt.plot(x,y,"ko-",linewidth=2,markersize=8)
#
# #Label corresponding to horizontal and vertical coordinates
# plt.ylabel("权重",front3)
# plt.xlabel("损失值",front3)
#
#
# plt.axis([0,0.6,0,1.1])                #plt.axis([xmin,xmax,ymin,ymax])
# ax = plt.gca()
# for lable in ax.get_xticklabels() + ax.get_yticklabels():
#     lable.set_fontsize(10)  # 刻度大小
#
# # plt.legend(loc="upper left")
# plt.savefig("hard.jpg",dpi = 600,bbox_inches="tight")
# plt.show()


# import matplotlib.pyplot as plt
# import numpy as np
#
# # x = [0, 0.5, 1, 1.5, 2.0, 2.5, 3, 3.5, 4, 4.5, 5]
# # y = [0,0.45,0.55,0.90,0.31,0.03,0.52,0.69,0.59,0.59]
# x = np.arange(0, 0.6, 0.05)
# y = ((np.exp(20*x)-1) / (np.exp(20*x)+1))
#
# import numpy as np
# import matplotlib.pyplot as plt
# import random
# import torch
#
# c = [0.0168, 0.1970, 0.1064, 0.0775, 0.0828, 0.0154, 0.0412, 0.1178, 0.0859,
#        0.1793, 0.1327, 0.1043, 0.1377, 0.2248, 0.0880, 0.0869, 0.0484,0.0189, 0.2643, 0.0962, 0.0665, 0.1055, 0.0201, 0.0733, 0.1569, 0.0832,
#         0.2178, 0.1882, 0.1841, 0.1570, 0.3463, 0.1118, 0.1254, 0.0609,0.0172, 0.1409, 0.1298, 0.0598, 0.0923, 0.0179, 0.0425, 0.1375, 0.1142,
#         0.1659, 0.1278, 0.1320, 0.1909, 0.2120, 0.0994, 0.0803, 0.0374,0.0153, 0.1835, 0.0690, 0.0688, 0.0799, 0.0173, 0.0527, 0.1034, 0.0755,
#         0.1760, 0.1203, 0.1074, 0.0734, 0.2114, 0.1041, 0.0858, 0.0428,0.0198, 0.2429, 0.1114, 0.0750, 0.1121, 0.0196, 0.0724, 0.1614, 0.1121,
#         0.2140, 0.1872, 0.1919, 0.1742, 0.3346, 0.1093, 0.1258, 0.0594,0.0085, 0.0044, 0.0469, 0.0217, 0.0180, 0.0119, 0.0059, 0.0049, 0.0248,
#         0.0099, 0.0060, 0.0095, 0.0257, 0.0233, 0.0632, 0.0058, 0.0288,0.0117, 0.0744, 0.0938, 0.0432, 0.0623, 0.0175, 0.0272, 0.0669, 0.0655,
#         0.0984, 0.0533, 0.0725, 0.1232, 0.1204, 0.0883, 0.0533, 0.0230,0.0085, 0.0486, 0.0558, 0.0396, 0.0397, 0.0135, 0.0155, 0.0320, 0.0276,
#         0.0628, 0.0308, 0.0392, 0.0415, 0.0736, 0.0735, 0.0263, 0.0201,0.0124, 0.1326, 0.0464, 0.0619, 0.0445, 0.0128, 0.0286, 0.0437, 0.0794,
#         0.0924, 0.0791, 0.0436, 0.0319, 0.1091, 0.0789, 0.0501, 0.0371,0.0177, 0.2505, 0.0901, 0.0666, 0.0846, 0.0190, 0.0697, 0.1368, 0.0983,
#         0.2222, 0.1712, 0.1673, 0.1470, 0.3295, 0.1095, 0.1200, 0.0536,0.0163, 0.1651, 0.1317, 0.0750, 0.0956, 0.0180, 0.0491, 0.1125, 0.1201,
#         0.1701, 0.1353, 0.1088, 0.1715, 0.1963, 0.0932, 0.0911, 0.0454,0.0123, 0.1429, 0.0765, 0.0531, 0.0506, 0.0142, 0.0297, 0.0462, 0.0580,
#         0.1201, 0.0732, 0.0608, 0.0677, 0.1367, 0.0864, 0.0517, 0.0490,0.0134, 0.1556, 0.0839, 0.0699, 0.0651, 0.0142, 0.0412, 0.0717, 0.0793,
#         0.1356, 0.0987, 0.0697, 0.0565, 0.1479, 0.0776, 0.0663, 0.0431,0.0176, 0.1754, 0.0879, 0.0541, 0.0632, 0.0170, 0.0571, 0.1051, 0.0720,
#         0.2029, 0.1188, 0.1432, 0.1380, 0.2802, 0.1000, 0.0951, 0.0362,0.0176, 0.1754, 0.0879, 0.0541, 0.0632, 0.0170, 0.0571, 0.1051, 0.0720,
#         0.2029, 0.1188, 0.1432, 0.1380, 0.2802, 0.1000, 0.0951, 0.0362,0.0134, 0.1171, 0.0797, 0.0531, 0.0556, 0.0149, 0.0377, 0.0704, 0.0651,
#         0.1322, 0.0797, 0.0842, 0.0985, 0.1482, 0.0964, 0.0667, 0.0302,0.0138, 0.0403, 0.1049, 0.0416, 0.0651, 0.0166, 0.0244, 0.0874, 0.0810,
#         0.0927, 0.0793, 0.0833, 0.1603, 0.1036, 0.0983, 0.0380, 0.0226,0.0191, 0.2035, 0.1218, 0.0723, 0.1088, 0.0194, 0.0625, 0.1540, 0.1135,
#         0.1883, 0.1562, 0.1564, 0.1901, 0.2611, 0.1048, 0.1094, 0.0515,0.0200, 0.2043, 0.1367, 0.0839, 0.1136, 0.0193, 0.0610, 0.1479, 0.1390,
#         0.1912, 0.1652, 0.1324, 0.1942, 0.2423, 0.1049, 0.1066, 0.0567,0.0177, 0.1366, 0.1390, 0.0576, 0.0909, 0.0179, 0.0409, 0.1251, 0.0993,
#         0.1593, 0.1303, 0.1335, 0.1794, 0.1946, 0.0984, 0.0749, 0.0351,0.0138, 0.1410, 0.0865, 0.0577, 0.0646, 0.0160, 0.0413, 0.0814, 0.0788,
#         0.1395, 0.0992, 0.0829, 0.0948, 0.1407, 0.0921, 0.0681, 0.0337,0.0124, 0.1274, 0.0544, 0.0632, 0.0378, 0.0124, 0.0229, 0.0394, 0.0749,
#         0.0794, 0.0716, 0.0402, 0.0272, 0.0933, 0.0706, 0.0439, 0.0351,0.0169, 0.1736, 0.0812, 0.0720, 0.0673, 0.0163, 0.0428, 0.0839, 0.0936,
#         0.1613, 0.1072, 0.0770, 0.0852, 0.1729, 0.0968, 0.0772, 0.0483,0.0202, 0.2249, 0.1061, 0.0719, 0.0965, 0.0181, 0.0641, 0.1683, 0.1050,
#         0.2186, 0.1827, 0.1601, 0.1611, 0.2846, 0.0968, 0.1191, 0.0596,0.0156, 0.1378, 0.1026, 0.0556, 0.0763, 0.0170, 0.0446, 0.1039, 0.0852,
#         0.1665, 0.1068, 0.1029, 0.1054, 0.1714, 0.0901, 0.0758, 0.0361,0.0106, 0.0389, 0.0565, 0.0329, 0.0459, 0.0135, 0.0189, 0.0530, 0.0169,
#         0.0840, 0.0476, 0.0556, 0.0542, 0.0829, 0.0839, 0.0252, 0.0198,0.0135, 0.1333, 0.0510, 0.0580, 0.0487, 0.0150, 0.0312, 0.0489, 0.0856,
#         0.1060, 0.0835, 0.0481, 0.0461, 0.1201, 0.0880, 0.0533, 0.0435,0.0097, 0.0836, 0.0570, 0.0455, 0.0314, 0.0144, 0.0220, 0.0316, 0.0769,
#         0.0628, 0.0502, 0.0353, 0.0335, 0.0650, 0.0955, 0.0367, 0.0309,0.0092, 0.0632, 0.0595, 0.0345, 0.0434, 0.0137, 0.0246, 0.0406, 0.0170,
#         0.0950, 0.0453, 0.0675, 0.0443, 0.1056, 0.0863, 0.0262, 0.0250,0.0174, 0.2286, 0.0991, 0.0532, 0.0789, 0.0179, 0.0726, 0.1303, 0.0840,
#         0.2243, 0.1409, 0.1890, 0.1665, 0.3649, 0.1072, 0.1122, 0.0451]
# c = c *50
# c = sorted(c)
# c = torch.tensor(c)
#
# yf1=(np.exp(1*c)-1)/(np.exp(1*c)+1)
# d1 = yf1 * c
#
# yf2=(np.exp(15*c)-1)/(np.exp(15*c)+1)
# d2 = yf2 * c
#
# yf3=(np.exp(30*c)-1)/(np.exp(30*c)+1)
# d3 = yf3 * c
#
# import matplotlib
# matplotlib.rcParams['xtick.direction'] = 'in'
# matplotlib.rcParams['ytick.direction'] = 'in'
# '''
# plt.plot(Abscissa, ordinate,"bo-",linewidth="",markersize="",label="")
# "bo-":The shape of the point of the curve
# linewidth:The distance from the point on the curve
# markersize:The size of the point on the curve
# label:Description of the curve
# '''
# # plt.plot(x,y,"ko-",linewidth=2,markersize=8)
# print(len(c))
# plt.plot(c, color="blue", linewidth=2.0, linestyle="-", label="MSE", alpha=0.9)  # 颜色 线宽 类型 标签 透明度
# plt.plot(d1, color="red", linewidth=2.0, linestyle="-", label="γ=1", alpha=0.9)
# plt.plot(d2, "green", linewidth=2.0, linestyle="-",label="γ=20")  # r*:red *  颜色为红色，线型为*
# plt.plot(d3, "purple", linewidth=2.0, linestyle="-",label="γ=30")
# #Label corresponding to horizontal and vertical coordinates
# plt.ylabel("损失值",front3)
# plt.xlabel("关键点索引",front3)
# # my_x_ticks = np.arange(0, 25501, 5000)
# # my_y_ticks = np.arange(0, 0.41, 0.1)
# # plt.xticks(my_x_ticks)
# # plt.yticks(my_y_ticks)
# plt.axis([0,25555,0,0.451])                #plt.axis([xmin,xmax,ymin,ymax])
#
# ax = plt.gca()
# for lable in ax.get_xticklabels() + ax.get_yticklabels():
#     lable.set_fontsize(10)  # 刻度大小
#
# plt.legend(loc="upper left")
# plt.savefig("hard.jpg",dpi = 600,bbox_inches="tight")
# plt.show()

# import matplotlib.pyplot as plt
# import numpy as np
# x =  np.arange(0, 32, 2)
# y = [75.1, 73.6, 74.0, 74.6, 75.0, 75.2, 75.3, 75.5, 75.8, 75.8, 75.9, 76.0, 76.4, 76.2, 76.1, 75.1]
# import matplotlib
# matplotlib.rcParams['xtick.direction'] = 'in'
# matplotlib.rcParams['ytick.direction'] = 'in'
# '''
# plt.plot(Abscissa, ordinate,"bo-",linewidth="",markersize="",label="")
# "bo-":The shape of the point of the curve
# linewidth:The distance from the point on the curve
# markersize:The size of the point on the curve
# label:Description of the curve
# '''
# plt.plot(x,y,"ko-",linewidth=2,markersize=8)
#
# #Label corresponding to horizontal and vertical coordinates
# plt.ylabel("mAP/%",front2)
# plt.xlabel("γ取值",front3)
#
# # plt.axis([0,32,70,80])                #plt.axis([xmin,xmax,ymin,ymax])
# my_x_ticks = np.arange(0, 32, 2)
# my_y_ticks = [0,73,74,75,76,77]
# plt.xticks(my_x_ticks)
# plt.yticks([71,73,74,75,76,77],['0','73','74','75','76','77'])
#
# ax = plt.gca()
# for lable in ax.get_xticklabels() + ax.get_yticklabels():
#     lable.set_fontsize(10)  # 刻度大小
#
# # plt.legend(loc="upper left")
# plt.savefig("d.jpg",dpi = 600,bbox_inches="tight")
# plt.show()
# --- End of .\plot.py ---

# --- Start of .\demo\inference.py ---
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import os
import shutil

from PIL import Image
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision
import cv2
import numpy as np

import sys
sys.path.append("../lib")
import time

# import _init_paths
import models
from config import cfg
from config import update_config
from core.inference import get_final_preds
from utils.transforms import get_affine_transform

CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


COCO_KEYPOINT_INDEXES = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
}

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def get_person_detection_boxes(model, img, threshold=0.5):
    pil_image = Image.fromarray(img)  # Load the image
    transform = transforms.Compose([transforms.ToTensor()])  # Defing PyTorch Transform
    transformed_img = transform(pil_image)  # Apply the transform to the image
    pred = model([transformed_img.to(CTX)])  # Pass the image to the model
    # Use the first detected person
    pred_classes = [COCO_INSTANCE_CATEGORY_NAMES[i]
                    for i in list(pred[0]['labels'].cpu().numpy())]  # Get the Prediction Score
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])]
                  for i in list(pred[0]['boxes'].cpu().detach().numpy())]  # Bounding boxes
    pred_scores = list(pred[0]['scores'].cpu().detach().numpy())

    person_boxes = []
    # Select box has score larger than threshold and is person
    for pred_class, pred_box, pred_score in zip(pred_classes, pred_boxes, pred_scores):
        if (pred_score > threshold) and (pred_class == 'person'):
            person_boxes.append(pred_box)

    return person_boxes


def get_pose_estimation_prediction(pose_model, image, centers, scales, transform):
    rotation = 0

    # pose estimation transformation
    model_inputs = []
    for center, scale in zip(centers, scales):
        trans = get_affine_transform(center, scale, rotation, cfg.MODEL.IMAGE_SIZE)
        # Crop smaller image of people
        model_input = cv2.warpAffine(
            image,
            trans,
            (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
            flags=cv2.INTER_LINEAR)

        # hwc -> 1chw
        model_input = transform(model_input)#.unsqueeze(0)
        model_inputs.append(model_input)

    # n * 1chw -> nchw
    model_inputs = torch.stack(model_inputs)

    # compute output heatmap
    output = pose_model(model_inputs.to(CTX))
    coords, _ = get_final_preds(
        cfg,
        output.cpu().detach().numpy(),
        np.asarray(centers),
        np.asarray(scales))

    return coords


def box_to_center_scale(box, model_image_width, model_image_height):
    """convert a box to center,scale information required for pose transformation
    Parameters
    ----------
    box : list of tuple
        list of length 2 with two tuples of floats representing
        bottom left and top right corner of a box
    model_image_width : int
    model_image_height : int

    Returns
    -------
    (numpy array, numpy array)
        Two numpy arrays, coordinates for the center of the box and the scale of the box
    """
    center = np.zeros((2), dtype=np.float32)

    bottom_left_corner = box[0]
    top_right_corner = box[1]
    box_width = top_right_corner[0]-bottom_left_corner[0]
    box_height = top_right_corner[1]-bottom_left_corner[1]
    bottom_left_x = bottom_left_corner[0]
    bottom_left_y = bottom_left_corner[1]
    center[0] = bottom_left_x + box_width * 0.5
    center[1] = bottom_left_y + box_height * 0.5

    aspect_ratio = model_image_width * 1.0 / model_image_height
    pixel_std = 200

    if box_width > aspect_ratio * box_height:
        box_height = box_width * 1.0 / aspect_ratio
    elif box_width < aspect_ratio * box_height:
        box_width = box_height * aspect_ratio
    scale = np.array(
        [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale


def prepare_output_dirs(prefix='/output/'):
    pose_dir = os.path.join(prefix, "pose")
    if os.path.exists(pose_dir) and os.path.isdir(pose_dir):
        shutil.rmtree(pose_dir)
    os.makedirs(pose_dir, exist_ok=True)
    return pose_dir


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--videoFile', type=str, required=True)
    parser.add_argument('--outputDir', type=str, default='/output/')
    parser.add_argument('--inferenceFps', type=int, default=10)
    parser.add_argument('--writeBoxFrames', action='store_true')

    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # args expected by supporting codebase
    args.modelDir = ''
    args.logDir = ''
    args.dataDir = ''
    args.prevModelDir = ''
    return args


def main():
    # transformation
    pose_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    args = parse_args()
    update_config(cfg, args)
    pose_dir = prepare_output_dirs(args.outputDir)
    csv_output_rows = []

    box_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    box_model.to(CTX)
    box_model.eval()
    pose_model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    if cfg.TEST.MODEL_FILE:
        print('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        pose_model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        print('expected model defined in config at TEST.MODEL_FILE')

    pose_model.to(CTX)
    pose_model.eval()

    # Loading an video
    vidcap = cv2.VideoCapture(args.videoFile)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    if fps < args.inferenceFps:
        print('desired inference fps is '+str(args.inferenceFps)+' but video fps is '+str(fps))
        exit()
    skip_frame_cnt = round(fps / args.inferenceFps)
    frame_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    outcap = cv2.VideoWriter('{}/{}_pose.avi'.format(args.outputDir, os.path.splitext(os.path.basename(args.videoFile))[0]),
                             cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), int(skip_frame_cnt), (frame_width, frame_height))

    count = 0
    while vidcap.isOpened():
        total_now = time.time()
        ret, image_bgr = vidcap.read()
        count += 1

        if not ret:
            continue

        if count % skip_frame_cnt != 0:
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Clone 2 image for person detection and pose estimation
        if cfg.DATASET.COLOR_RGB:
            image_per = image_rgb.copy()
            image_pose = image_rgb.copy()
        else:
            image_per = image_bgr.copy()
            image_pose = image_bgr.copy()

        # Clone 1 image for debugging purpose
        image_debug = image_bgr.copy()

        # object detection box
        now = time.time()
        pred_boxes = get_person_detection_boxes(box_model, image_per, threshold=0.9)
        then = time.time()
        print("Find person bbox in: {} sec".format(then - now))

        # Can not find people. Move to next frame
        if not pred_boxes:
            count += 1
            continue

        if args.writeBoxFrames:
            for box in pred_boxes:
                cv2.rectangle(image_debug, box[0], box[1], color=(0, 255, 0),
                              thickness=3)  # Draw Rectangle with the coordinates

        # pose estimation : for multiple people
        centers = []
        scales = []
        for box in pred_boxes:
            center, scale = box_to_center_scale(box, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
            centers.append(center)
            scales.append(scale)

        now = time.time()
        pose_preds = get_pose_estimation_prediction(pose_model, image_pose, centers, scales, transform=pose_transform)
        then = time.time()
        print("Find person pose in: {} sec".format(then - now))

        new_csv_row = []
        for coords in pose_preds:
            # Draw each point on image
            for coord in coords:
                x_coord, y_coord = int(coord[0]), int(coord[1])
                cv2.circle(image_debug, (x_coord, y_coord), 4, (255, 0, 0), 2)
                new_csv_row.extend([x_coord, y_coord])

        total_then = time.time()

        text = "{:03.2f} sec".format(total_then - total_now)
        cv2.putText(image_debug, text, (100, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("pos", image_debug)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        csv_output_rows.append(new_csv_row)
        img_file = os.path.join(pose_dir, 'pose_{:08d}.jpg'.format(count))
        cv2.imwrite(img_file, image_debug)
        outcap.write(image_debug)


    # write csv
    csv_headers = ['frame']
    for keypoint in COCO_KEYPOINT_INDEXES.values():
        csv_headers.extend([keypoint+'_x', keypoint+'_y'])

    csv_output_filename = os.path.join(args.outputDir, 'pose-data.csv')
    with open(csv_output_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(csv_headers)
        csvwriter.writerows(csv_output_rows)

    vidcap.release()
    outcap.release()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

# --- End of .\demo\inference.py ---

# --- Start of .\lib\config\default.py ---

# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN


_C = CN()

_C.OUTPUT_DIR = ''
_C.LOG_DIR = ''
_C.DATA_DIR = ''
_C.GPUS = (0,)
_C.WORKERS = 4
_C.PRINT_FREQ = 20
_C.AUTO_RESUME = False
_C.PIN_MEMORY = True
_C.RANK = 0

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'pose_hrnet'
_C.MODEL.INIT_WEIGHTS = True
_C.MODEL.PRETRAINED = ''
_C.MODEL.NUM_JOINTS = 17
_C.MODEL.TAG_PER_JOINT = True
_C.MODEL.TARGET_TYPE = 'gaussian'
_C.MODEL.IMAGE_SIZE = [256, 256]  # width * height, ex: 192 * 256
_C.MODEL.HEATMAP_SIZE = [64, 64]  # width * height, ex: 24 * 32
_C.MODEL.SIGMA = 2
_C.MODEL.EXTRA = CN(new_allowed=True)

_C.LOSS = CN()
_C.LOSS.USE_OHKM = False
_C.LOSS.TOPK = 8
_C.LOSS.USE_TARGET_WEIGHT = True
_C.LOSS.USE_DIFFERENT_JOINTS_WEIGHT = False

# DATASET related params
_C.DATASET = CN()
_C.DATASET.ROOT = ''
_C.DATASET.DATASET = 'mpii'
_C.DATASET.TRAIN_SET = 'train'
_C.DATASET.TEST_SET = 'valid'
_C.DATASET.DATA_FORMAT = 'jpg'
_C.DATASET.HYBRID_JOINTS_TYPE = ''
_C.DATASET.SELECT_DATA = False

# training data augmentation
_C.DATASET.FLIP = True
_C.DATASET.SCALE_FACTOR = 0.25
_C.DATASET.ROT_FACTOR = 30
_C.DATASET.PROB_HALF_BODY = 0.0
_C.DATASET.NUM_JOINTS_HALF_BODY = 8
_C.DATASET.COLOR_RGB = False

# train
_C.TRAIN = CN()

_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP = [90, 110]
_C.TRAIN.LR = 0.001

_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 0.0001
_C.TRAIN.NESTEROV = False
_C.TRAIN.GAMMA1 = 0.99
_C.TRAIN.GAMMA2 = 0.0

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 140

_C.TRAIN.RESUME = False
_C.TRAIN.CHECKPOINT = ''

_C.TRAIN.BATCH_SIZE_PER_GPU = 32
_C.TRAIN.SHUFFLE = True

# testing
_C.TEST = CN()

# size of images for each device
_C.TEST.BATCH_SIZE_PER_GPU = 32
# Test Model Epoch
_C.TEST.FLIP_TEST = False
_C.TEST.POST_PROCESS = False
_C.TEST.SHIFT_HEATMAP = False

_C.TEST.USE_GT_BBOX = False

# nms
_C.TEST.IMAGE_THRE = 0.1
_C.TEST.NMS_THRE = 0.6
_C.TEST.SOFT_NMS = False
_C.TEST.OKS_THRE = 0.5
_C.TEST.IN_VIS_THRE = 0.0
_C.TEST.COCO_BBOX_FILE = ''
_C.TEST.BBOX_THRE = 1.0
_C.TEST.MODEL_FILE = ''

# debug
_C.DEBUG = CN()
_C.DEBUG.DEBUG = False
_C.DEBUG.SAVE_BATCH_IMAGES_GT = False
_C.DEBUG.SAVE_BATCH_IMAGES_PRED = False
_C.DEBUG.SAVE_HEATMAPS_GT = False
_C.DEBUG.SAVE_HEATMAPS_PRED = False


def update_config(cfg, args):
    cfg.defrost()
    # 对于不同的模型配置，你有不同的超参设置，所以你可以使用yaml文件来管理不同的configs，
    # 然后使用merge_from_file()这个方法，这个会比较每个模型特有的config和默认参数的区别，
    # 会将默认参数与特定参数不同的部分，用特定参数覆盖
    cfg.merge_from_file(args.cfg)
    # 对于不同的模型配置，你有不同的超参设置，所以你可以使用列表来管理不同的configs，
    # 然后使用merge_from_list()这个方法，这个会比较每个模型特有的config和默认参数的区别，
    # 会将默认参数与特定参数不同的部分，用特定参数覆盖
    cfg.merge_from_list(args.opts)

    if args.modelDir:
        cfg.OUTPUT_DIR = args.modelDir

    if args.logDir:
        cfg.LOG_DIR = args.logDir

    if args.dataDir:
        cfg.DATA_DIR = args.dataDir

    cfg.DATASET.ROOT = os.path.join(
        cfg.DATA_DIR, cfg.DATASET.ROOT
    )

    cfg.MODEL.PRETRAINED = os.path.join(
        cfg.DATA_DIR, cfg.MODEL.PRETRAINED
    )

    if cfg.TEST.MODEL_FILE:
        cfg.TEST.MODEL_FILE = os.path.join(
            cfg.DATA_DIR, cfg.TEST.MODEL_FILE
        )

    cfg.freeze()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)


# --- End of .\lib\config\default.py ---

# --- Start of .\lib\config\models.py ---
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yacs.config import CfgNode as CN


# pose_resnet related params
POSE_RESNET = CN()
POSE_RESNET.NUM_LAYERS = 50
POSE_RESNET.DECONV_WITH_BIAS = False
POSE_RESNET.NUM_DECONV_LAYERS = 3
POSE_RESNET.NUM_DECONV_FILTERS = [256, 256, 256]
POSE_RESNET.NUM_DECONV_KERNELS = [4, 4, 4]
POSE_RESNET.FINAL_CONV_KERNEL = 1
POSE_RESNET.PRETRAINED_LAYERS = ['*']

# pose_multi_resoluton_net related params
POSE_HIGH_RESOLUTION_NET = CN()
POSE_HIGH_RESOLUTION_NET.PRETRAINED_LAYERS = ['*']
POSE_HIGH_RESOLUTION_NET.STEM_INPLANES = 64
POSE_HIGH_RESOLUTION_NET.FINAL_CONV_KERNEL = 1

POSE_HIGH_RESOLUTION_NET.STAGE2 = CN()
POSE_HIGH_RESOLUTION_NET.STAGE2.NUM_MODULES = 1
POSE_HIGH_RESOLUTION_NET.STAGE2.NUM_BRANCHES = 2
POSE_HIGH_RESOLUTION_NET.STAGE2.NUM_BLOCKS = [4, 4]
POSE_HIGH_RESOLUTION_NET.STAGE2.NUM_CHANNELS = [32, 64]
POSE_HIGH_RESOLUTION_NET.STAGE2.BLOCK = 'BASIC'
POSE_HIGH_RESOLUTION_NET.STAGE2.FUSE_METHOD = 'SUM'

POSE_HIGH_RESOLUTION_NET.STAGE3 = CN()
POSE_HIGH_RESOLUTION_NET.STAGE3.NUM_MODULES = 1
POSE_HIGH_RESOLUTION_NET.STAGE3.NUM_BRANCHES = 3
POSE_HIGH_RESOLUTION_NET.STAGE3.NUM_BLOCKS = [4, 4, 4]
POSE_HIGH_RESOLUTION_NET.STAGE3.NUM_CHANNELS = [32, 64, 128]
POSE_HIGH_RESOLUTION_NET.STAGE3.BLOCK = 'BASIC'
POSE_HIGH_RESOLUTION_NET.STAGE3.FUSE_METHOD = 'SUM'

POSE_HIGH_RESOLUTION_NET.STAGE4 = CN()
POSE_HIGH_RESOLUTION_NET.STAGE4.NUM_MODULES = 1
POSE_HIGH_RESOLUTION_NET.STAGE4.NUM_BRANCHES = 4
POSE_HIGH_RESOLUTION_NET.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
POSE_HIGH_RESOLUTION_NET.STAGE4.NUM_CHANNELS = [32, 64, 128, 256]
POSE_HIGH_RESOLUTION_NET.STAGE4.BLOCK = 'BASIC'
POSE_HIGH_RESOLUTION_NET.STAGE4.FUSE_METHOD = 'SUM'


MODEL_EXTRAS = {
    'pose_resnet': POSE_RESNET,
    'pose_high_resolution_net': POSE_HIGH_RESOLUTION_NET,
}

# --- End of .\lib\config\models.py ---

# --- Start of .\lib\config\__init__.py ---
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from .default import _C as cfg
from .default import update_config
from .models import MODEL_EXTRAS

# --- End of .\lib\config\__init__.py ---

# --- Start of .\lib\core\evaluate.py ---
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from core.inference import get_max_preds


def calc_dists(preds, target, normalize):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
            else:
                dists[c, n] = -1
    return dists


def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1


def accuracy(output, target, hm_type='gaussian', thr=0.5):
    '''
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    '''
    idx = list(range(output.shape[1]))
    norm = 1.0
    if hm_type == 'gaussian':
        pred, _ = get_max_preds(output)
        target, _ = get_max_preds(target)
        h = output.shape[2]
        w = output.shape[3]
        norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10
    dists = calc_dists(pred, target, norm)

    acc = np.zeros((len(idx) + 1))
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i + 1] = dist_acc(dists[idx[i]])
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc
    return acc, avg_acc, cnt, pred



# --- End of .\lib\core\evaluate.py ---

# --- Start of .\lib\core\function.py ---
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import os

import numpy as np
import torch

from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.transforms import flip_back
from utils.vis import save_debug_images

logger = logging.getLogger(__name__)


def train(config, train_loader, model, model_fine, criterion, criterion_fine, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()
    model_fine.train()

    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        outputs = model(input)
        outputs_fine = model_fine(outputs)  # gaidong

        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)
        if isinstance(outputs, list):
            loss = criterion(outputs[0], target, target_weight)
            for output in outputs[1:]:
                loss += criterion(output, target, target_weight)
        else:
            output = outputs
            loss = criterion(output, target, target_weight)
        if isinstance(outputs_fine, list):
            loss_fine = criterion_fine(outputs_fine[0], target, target_weight)
            for output_fine in outputs_fine[1:]:
                loss_fine += criterion_fine(output_fine, target, target_weight)
        else:
            output_fine = outputs_fine
            loss_fine = criterion_fine(output_fine, target, target_weight)

        loss_toal = loss + loss_fine
        # compute gradient and do update step
        optimizer.zero_grad()
        loss_toal.backward()
        optimizer.step()

        # Free CUDA memory after each batch
        torch.cuda.empty_cache()

        # measure accuracy and record loss
        losses.update(loss_toal.item(), input.size(0))
        _, avg_acc, cnt, pred = accuracy(output_fine.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                speed=input.size(0) / batch_time.val,
                data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred * 4, output,
                              prefix)


def validate(config, val_loader, val_dataset, model, model_fine, criterion, criterion_fine, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()
    model_fine.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            # compute output
            outputs = model(input)
            outputs = model_fine(outputs)
            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs

            if config.TEST.FLIP_TEST:
                input_flipped = input.flip(3)
                outputs_flipped = model(input_flipped)

                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[-1]
                else:
                    output_flipped = outputs_flipped

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            loss = criterion(output, target, target_weight)
            loss_fine = criterion_fine(output, target, target_weight)
            loss_total = loss + loss_fine
            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss_total.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s * 200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time,
                    loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, target, pred * 4, output,
                                  prefix)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values + 1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
        ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

# --- End of .\lib\core\function.py ---

# --- Start of .\lib\core\inference.py ---
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np

from utils.transforms import transform_preds


def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def get_final_preds(config, batch_heatmaps, center, scale):
    coords, maxvals = get_max_preds(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    if config.TEST.POST_PROCESS:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                    diff = np.array(
                        [
                            hm[py][px+1] - hm[py][px-1],
                            hm[py+1][px]-hm[py-1][px]
                        ]
                    )
                    coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(
            coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
        )

    return preds, maxvals

# --- End of .\lib\core\inference.py ---

# --- Start of .\lib\core\loss.py ---
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints


class JointsOHKMMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsOHKMMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight

    def ohkm(self, loss):
        ohkm_loss = 0.
        for i in range(loss.size()[0]):
            sub_loss = loss[i]
            average_loss = torch.mean(sub_loss,dim=[0])
            yt = 1/(1+torch.exp(150*(-sub_loss+average_loss)))
            y = yt * sub_loss
            ohkm_loss += torch.mean(y)
        ohkm_loss /= loss.size()[0]
        return ohkm_loss

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = []
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss.append(0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                ))
            else:
                loss.append(
                    0.5 * self.criterion(heatmap_pred, heatmap_gt)
                )

        loss = [l.mean(dim=1).unsqueeze(dim=1) for l in loss]
        loss = torch.cat(loss, dim=1)
        return self.ohkm(loss)

# --- End of .\lib\core\loss.py ---

# --- Start of .\lib\dataset\coco.py ---
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
from collections import OrderedDict
import logging
import os

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json_tricks as json
import numpy as np

from dataset.JointsDataset import JointsDataset
from nms.nms import oks_nms
from nms.nms import soft_oks_nms


logger = logging.getLogger(__name__)


class COCODataset(JointsDataset):
    '''
    "keypoints": {
        0: "nose",
        1: "left_eye",
        2: "right_eye",
        3: "left_ear",
        4: "right_ear",
        5: "left_shoulder",
        6: "right_shoulder",
        7: "left_elbow",
        8: "right_elbow",
        9: "left_wrist",
        10: "right_wrist",
        11: "left_hip",
        12: "right_hip",
        13: "left_knee",
        14: "right_knee",
        15: "left_ankle",
        16: "right_ankle"
    },
	"skeleton": [
        [16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13], [6,7],[6,8],
        [7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]
    '''
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        super().__init__(cfg, root, image_set, is_train, transform)
        self.nms_thre = cfg.TEST.NMS_THRE
        self.image_thre = cfg.TEST.IMAGE_THRE
        self.soft_nms = cfg.TEST.SOFT_NMS
        self.oks_thre = cfg.TEST.OKS_THRE
        self.in_vis_thre = cfg.TEST.IN_VIS_THRE
        self.bbox_file = cfg.TEST.COCO_BBOX_FILE
        self.use_gt_bbox = cfg.TEST.USE_GT_BBOX
        self.image_width = cfg.MODEL.IMAGE_SIZE[0]
        self.image_height = cfg.MODEL.IMAGE_SIZE[1]
        self.aspect_ratio = self.image_width * 1.0 / self.image_height
        self.pixel_std = 200

        self.coco = COCO(self._get_ann_file_keypoint())

        # deal with class names
        cats = [cat['name']
                for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.classes = ['__background__'] + cats
        logger.info('=> classes: {}'.format(self.classes))
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        self._coco_ind_to_class_ind = dict(
            [
                (self._class_to_coco_ind[cls], self._class_to_ind[cls])
                for cls in self.classes[1:]
            ]
        )

        # load image file names
        self.image_set_index = self._load_image_set_index()
        self.num_images = len(self.image_set_index)
        logger.info('=> num_images: {}'.format(self.num_images))

        self.num_joints = 17
        self.flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8],
                           [9, 10], [11, 12], [13, 14], [15, 16]]
        self.parent_ids = None
        self.upper_body_ids = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        self.lower_body_ids = (11, 12, 13, 14, 15, 16)

        self.joints_weight = np.array(
            [
                1., 1., 1., 1., 1., 1., 1., 1.2, 1.2,
                1.5, 1.5, 1., 1., 1.2, 1.2, 1.5, 1.5
            ],
            dtype=np.float32
        ).reshape((self.num_joints, 1))

        self.db = self._get_db()

        if is_train and cfg.DATASET.SELECT_DATA:
            self.db = self.select_data(self.db)

        logger.info('=> load {} samples'.format(len(self.db)))

    def _get_ann_file_keypoint(self):
        """ self.root / annotations / person_keypoints_train2017.json """
        prefix = 'person_keypoints' \
            if 'test' not in self.image_set else 'image_info'
        return os.path.join(
            self.root,
            'annotations',
            prefix + '_' + self.image_set + '.json'
        )

    def _load_image_set_index(self):
        """ image id: int """
        image_ids = self.coco.getImgIds()
        return image_ids

    def _get_db(self):
        if self.is_train or self.use_gt_bbox:
            # use ground truth bbox
            gt_db = self._load_coco_keypoint_annotations()
        else:
            # use bbox from detection
            gt_db = self._load_coco_person_detection_results()
        return gt_db

    def _load_coco_keypoint_annotations(self):
        """ ground truth bbox and keypoints """
        gt_db = []
        for index in self.image_set_index:
            gt_db.extend(self._load_coco_keypoint_annotation_kernal(index))
        return gt_db

    def _load_coco_keypoint_annotation_kernal(self, index):
        """
        coco ann: [u'segmentation', u'area', u'iscrowd', u'image_id', u'bbox', u'category_id', u'id']
        iscrowd:
            crowd instances are handled by marking their overlaps with all categories to -1
            and later excluded in training
        bbox:
            [x1, y1, w, h]
        :param index: coco image id
        :return: db entry
        """
        im_ann = self.coco.loadImgs(index)[0]
        width = im_ann['width']
        height = im_ann['height']

        annIds = self.coco.getAnnIds(imgIds=index, iscrowd=False)
        objs = self.coco.loadAnns(annIds)

        # sanitize bboxes
        valid_objs = []
        for obj in objs:
            x, y, w, h = obj['bbox']
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2-x1, y2-y1]
                valid_objs.append(obj)
        objs = valid_objs

        rec = []
        for obj in objs:
            cls = self._coco_ind_to_class_ind[obj['category_id']]
            if cls != 1:
                continue

            # ignore objs without keypoints annotation
            if max(obj['keypoints']) == 0:
                continue

            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float64)
            joints_3d_vis = np.zeros((self.num_joints, 3), dtype=np.float64)
            for ipt in range(self.num_joints):
                joints_3d[ipt, 0] = obj['keypoints'][ipt * 3 + 0]
                joints_3d[ipt, 1] = obj['keypoints'][ipt * 3 + 1]
                joints_3d[ipt, 2] = 0
                t_vis = obj['keypoints'][ipt * 3 + 2]
                if t_vis > 1:
                    t_vis = 1
                joints_3d_vis[ipt, 0] = t_vis
                joints_3d_vis[ipt, 1] = t_vis
                joints_3d_vis[ipt, 2] = 0

            center, scale = self._box2cs(obj['clean_bbox'][:4])
            rec.append({
                'image': self.image_path_from_index(index),
                'center': center,
                'scale': scale,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
                'filename': '',
                'imgnum': 0,
            })

        return rec

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale

    def image_path_from_index(self, index):
        """ example: images / train2017 / 000000119993.jpg """
        file_name = '%012d.jpg' % index
        if '2014' in self.image_set:
            file_name = 'COCO_%s_' % self.image_set + file_name

        prefix = 'test2017' if 'test' in self.image_set else self.image_set

        data_name = prefix + '.zip@' if self.data_format == 'zip' else prefix

        image_path = os.path.join(
            self.root, 'images', data_name, file_name)

        return image_path

    def _load_coco_person_detection_results(self):
        all_boxes = None
        with open(self.bbox_file, 'r') as f:
            all_boxes = json.load(f)

        if not all_boxes:
            logger.error('=> Load %s fail!' % self.bbox_file)
            return None

        logger.info('=> Total boxes: {}'.format(len(all_boxes)))

        kpt_db = []
        num_boxes = 0
        for n_img in range(0, len(all_boxes)):
            det_res = all_boxes[n_img]
            if det_res['category_id'] != 1:
                continue
            img_name = self.image_path_from_index(det_res['image_id'])
            box = det_res['bbox']
            score = det_res['score']

            if score < self.image_thre:
                continue

            num_boxes = num_boxes + 1

            center, scale = self._box2cs(box)
            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float64)
            joints_3d_vis = np.ones(
                (self.num_joints, 3), dtype=np.float64)
            kpt_db.append({
                'image': img_name,
                'center': center,
                'scale': scale,
                'score': score,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
            })

        logger.info('=> Total boxes after fliter low score@{}: {}'.format(
            self.image_thre, num_boxes))
        return kpt_db

    def evaluate(self, cfg, preds, output_dir, all_boxes, img_path,
                 *args, **kwargs):
        rank = cfg.RANK

        res_folder = os.path.join(output_dir, 'results')
        if not os.path.exists(res_folder):
            try:
                os.makedirs(res_folder)
            except Exception:
                logger.error('Fail to make {}'.format(res_folder))

        res_file = os.path.join(
            res_folder, 'keypoints_{}_results_{}.json'.format(
                self.image_set, rank)
        )

        # person x (keypoints)
        _kpts = []
        for idx, kpt in enumerate(preds):
            _kpts.append({
                'keypoints': kpt,
                'center': all_boxes[idx][0:2],
                'scale': all_boxes[idx][2:4],
                'area': all_boxes[idx][4],
                'score': all_boxes[idx][5],
                'image': int(img_path[idx][-16:-4])
            })
        # image x person x (keypoints)
        kpts = defaultdict(list)
        for kpt in _kpts:
            kpts[kpt['image']].append(kpt)

        # rescoring and oks nms
        num_joints = self.num_joints
        in_vis_thre = self.in_vis_thre
        oks_thre = self.oks_thre
        oks_nmsed_kpts = []
        for img in kpts.keys():
            img_kpts = kpts[img]
            for n_p in img_kpts:
                box_score = n_p['score']
                kpt_score = 0
                valid_num = 0
                for n_jt in range(0, num_joints):
                    t_s = n_p['keypoints'][n_jt][2]
                    if t_s > in_vis_thre:
                        kpt_score = kpt_score + t_s
                        valid_num = valid_num + 1
                if valid_num != 0:
                    kpt_score = kpt_score / valid_num
                # rescoring
                n_p['score'] = kpt_score * box_score

            if self.soft_nms:
                keep = soft_oks_nms(
                    [img_kpts[i] for i in range(len(img_kpts))],
                    oks_thre
                )
            else:
                keep = oks_nms(
                    [img_kpts[i] for i in range(len(img_kpts))],
                    oks_thre
                )

            if len(keep) == 0:
                oks_nmsed_kpts.append(img_kpts)
            else:
                oks_nmsed_kpts.append([img_kpts[_keep] for _keep in keep])

        self._write_coco_keypoint_results(
            oks_nmsed_kpts, res_file)
        if 'test' not in self.image_set:
            info_str = self._do_python_keypoint_eval(
                res_file, res_folder)
            name_value = OrderedDict(info_str)
            return name_value, name_value['AP']
        else:
            return {'Null': 0}, 0

    def _write_coco_keypoint_results(self, keypoints, res_file):
        data_pack = [
            {
                'cat_id': self._class_to_coco_ind[cls],
                'cls_ind': cls_ind,
                'cls': cls,
                'ann_type': 'keypoints',
                'keypoints': keypoints
            }
            for cls_ind, cls in enumerate(self.classes) if not cls == '__background__'
        ]

        results = self._coco_keypoint_results_one_category_kernel(data_pack[0])
        logger.info('=> writing results json to %s' % res_file)
        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)
        try:
            json.load(open(res_file))
        except Exception:
            content = []
            with open(res_file, 'r') as f:
                for line in f:
                    content.append(line)
            content[-1] = ']'
            with open(res_file, 'w') as f:
                for c in content:
                    f.write(c)

    def _coco_keypoint_results_one_category_kernel(self, data_pack):
        cat_id = data_pack['cat_id']
        keypoints = data_pack['keypoints']
        cat_results = []

        for img_kpts in keypoints:
            if len(img_kpts) == 0:
                continue

            _key_points = np.array([img_kpts[k]['keypoints']
                                    for k in range(len(img_kpts))])
            key_points = np.zeros(
                (_key_points.shape[0], self.num_joints * 3), dtype=np.float64
            )

            for ipt in range(self.num_joints):
                key_points[:, ipt * 3 + 0] = _key_points[:, ipt, 0]
                key_points[:, ipt * 3 + 1] = _key_points[:, ipt, 1]
                key_points[:, ipt * 3 + 2] = _key_points[:, ipt, 2]  # keypoints score.

            result = [
                {
                    'image_id': img_kpts[k]['image'],
                    'category_id': cat_id,
                    'keypoints': list(key_points[k]),
                    'score': img_kpts[k]['score'],
                    'center': list(img_kpts[k]['center']),
                    'scale': list(img_kpts[k]['scale'])
                }
                for k in range(len(img_kpts))
            ]
            cat_results.extend(result)

        return cat_results

    def _do_python_keypoint_eval(self, res_file, res_folder):
        coco_dt = self.coco.loadRes(res_file)
        coco_eval = COCOeval(self.coco, coco_dt, 'keypoints')
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats_names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']

        info_str = []
        for ind, name in enumerate(stats_names):
            info_str.append((name, coco_eval.stats[ind]))

        return info_str

# --- End of .\lib\dataset\coco.py ---

# --- Start of .\lib\dataset\JointsDataset.py ---
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
from utils.transforms import fliplr_joints


logger = logging.getLogger(__name__)


class JointsDataset(Dataset):
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
        self.num_joints_half_body = cfg.DATASET.NUM_JOINTS_HALF_BODY
        self.prob_half_body = cfg.DATASET.PROB_HALF_BODY
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

    def half_body_transform(self, joints, joints_vis):
        upper_joints = []
        lower_joints = []
        for joint_id in range(self.num_joints):
            if joints_vis[joint_id][0] > 0:
                if joint_id in self.upper_body_ids:
                    upper_joints.append(joints[joint_id])
                else:
                    lower_joints.append(joints[joint_id])

        if np.random.randn() < 0.5 and len(upper_joints) > 2:
            selected_joints = upper_joints
        else:
            selected_joints = lower_joints \
                if len(lower_joints) > 2 else upper_joints

        if len(selected_joints) < 2:
            return None, None

        selected_joints = np.array(selected_joints, dtype=np.float32)
        center = selected_joints.mean(axis=0)[:2]

        left_top = np.amin(selected_joints, axis=0)
        right_bottom = np.amax(selected_joints, axis=0)

        w = right_bottom[0] - left_top[0]
        h = right_bottom[1] - left_top[1]

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio

        scale = np.array(
            [
                w * 1.0 / self.pixel_std,
                h * 1.0 / self.pixel_std
            ],
            dtype=np.float32
        )

        scale = scale * 1.5

        return center, scale

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
            if (np.sum(joints_vis[:, 0]) > self.num_joints_half_body
                and np.random.rand() < self.prob_half_body):
                c_half_body, s_half_body = self.half_body_transform(
                    joints, joints_vis
                )

                if c_half_body is not None and s_half_body is not None:
                    c, s = c_half_body, s_half_body

            sf = self.scale_factor
            rf = self.rotation_factor
            s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
                if random.random() <= 0.6 else 0

            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                joints, joints_vis = fliplr_joints(
                    joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                c[0] = data_numpy.shape[1] - c[0] - 1

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

# --- End of .\lib\dataset\JointsDataset.py ---

# --- Start of .\lib\dataset\mpii.py ---
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import json_tricks as json
from collections import OrderedDict

import numpy as np
from scipy.io import loadmat, savemat

from dataset.JointsDataset import JointsDataset


logger = logging.getLogger(__name__)


class MPIIDataset(JointsDataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        super().__init__(cfg, root, image_set, is_train, transform)

        self.num_joints = 16
        self.flip_pairs = [[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]]
        self.parent_ids = [1, 2, 6, 6, 3, 4, 6, 6, 7, 8, 11, 12, 7, 7, 13, 14]

        self.upper_body_ids = (7, 8, 9, 10, 11, 12, 13, 14, 15)
        self.lower_body_ids = (0, 1, 2, 3, 4, 5, 6)

        self.db = self._get_db()

        if is_train and cfg.DATASET.SELECT_DATA:
            self.db = self.select_data(self.db)

        logger.info('=> load {} samples'.format(len(self.db)))

    def _get_db(self):
        # create train/val split
        file_name = os.path.join(
            self.root, 'annot', self.image_set+'.json'
        )
        with open(file_name) as anno_file:
            anno = json.load(anno_file)

        gt_db = []
        for a in anno:
            image_name = a['image']

            c = np.array(a['center'], dtype=np.float)
            s = np.array([a['scale'], a['scale']], dtype=np.float)

            # Adjust center/scale slightly to avoid cropping limbs
            if c[0] != -1:
                c[1] = c[1] + 15 * s[1]
                s = s * 1.25

            # MPII uses matlab format, index is based 1,
            # we should first convert to 0-based index
            c = c - 1

            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.zeros((self.num_joints,  3), dtype=np.float)
            if self.image_set != 'test':
                joints = np.array(a['joints'])
                joints[:, 0:2] = joints[:, 0:2] - 1
                joints_vis = np.array(a['joints_vis'])
                assert len(joints) == self.num_joints, \
                    'joint num diff: {} vs {}'.format(len(joints),
                                                      self.num_joints)

                joints_3d[:, 0:2] = joints[:, 0:2]
                joints_3d_vis[:, 0] = joints_vis[:]
                joints_3d_vis[:, 1] = joints_vis[:]

            image_dir = 'images.zip@' if self.data_format == 'zip' else 'images'
            gt_db.append(
                {
                    'image': os.path.join(self.root, image_dir, image_name),
                    'center': c,
                    'scale': s,
                    'joints_3d': joints_3d,
                    'joints_3d_vis': joints_3d_vis,
                    'filename': '',
                    'imgnum': 0,
                }
            )

        return gt_db

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        # convert 0-based index to 1-based index
        preds = preds[:, :, 0:2] + 1.0

        if output_dir:
            pred_file = os.path.join(output_dir, 'pred.mat')
            savemat(pred_file, mdict={'preds': preds})

        if 'test' in cfg.DATASET.TEST_SET:
            return {'Null': 0.0}, 0.0

        SC_BIAS = 0.6
        threshold = 0.5

        gt_file = os.path.join(cfg.DATASET.ROOT,
                               'annot',
                               'gt_{}.mat'.format(cfg.DATASET.TEST_SET))
        gt_dict = loadmat(gt_file)
        dataset_joints = gt_dict['dataset_joints']
        jnt_missing = gt_dict['jnt_missing']
        pos_gt_src = gt_dict['pos_gt_src']
        headboxes_src = gt_dict['headboxes_src']

        pos_pred_src = np.transpose(preds, [1, 2, 0])

        head = np.where(dataset_joints == 'head')[1][0]
        lsho = np.where(dataset_joints == 'lsho')[1][0]
        lelb = np.where(dataset_joints == 'lelb')[1][0]
        lwri = np.where(dataset_joints == 'lwri')[1][0]
        lhip = np.where(dataset_joints == 'lhip')[1][0]
        lkne = np.where(dataset_joints == 'lkne')[1][0]
        lank = np.where(dataset_joints == 'lank')[1][0]

        rsho = np.where(dataset_joints == 'rsho')[1][0]
        relb = np.where(dataset_joints == 'relb')[1][0]
        rwri = np.where(dataset_joints == 'rwri')[1][0]
        rkne = np.where(dataset_joints == 'rkne')[1][0]
        rank = np.where(dataset_joints == 'rank')[1][0]
        rhip = np.where(dataset_joints == 'rhip')[1][0]

        jnt_visible = 1 - jnt_missing
        uv_error = pos_pred_src - pos_gt_src
        uv_err = np.linalg.norm(uv_error, axis=1)
        headsizes = headboxes_src[1, :, :] - headboxes_src[0, :, :]
        headsizes = np.linalg.norm(headsizes, axis=0)
        headsizes *= SC_BIAS
        scale = np.multiply(headsizes, np.ones((len(uv_err), 1)))
        scaled_uv_err = np.divide(uv_err, scale)
        scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)
        jnt_count = np.sum(jnt_visible, axis=1)
        less_than_threshold = np.multiply((scaled_uv_err <= threshold),
                                          jnt_visible)
        PCKh = np.divide(100.*np.sum(less_than_threshold, axis=1), jnt_count)

        # save
        rng = np.arange(0, 0.5+0.01, 0.01)
        pckAll = np.zeros((len(rng), 16))

        for r in range(len(rng)):
            threshold = rng[r]
            less_than_threshold = np.multiply(scaled_uv_err <= threshold,
                                              jnt_visible)
            pckAll[r, :] = np.divide(100.*np.sum(less_than_threshold, axis=1),
                                     jnt_count)

        PCKh = np.ma.array(PCKh, mask=False)
        PCKh.mask[6:8] = True

        jnt_count = np.ma.array(jnt_count, mask=False)
        jnt_count.mask[6:8] = True
        jnt_ratio = jnt_count / np.sum(jnt_count).astype(np.float64)

        name_value = [
            ('Head', PCKh[head]),
            ('Shoulder', 0.5 * (PCKh[lsho] + PCKh[rsho])),
            ('Elbow', 0.5 * (PCKh[lelb] + PCKh[relb])),
            ('Wrist', 0.5 * (PCKh[lwri] + PCKh[rwri])),
            ('Hip', 0.5 * (PCKh[lhip] + PCKh[rhip])),
            ('Knee', 0.5 * (PCKh[lkne] + PCKh[rkne])),
            ('Ankle', 0.5 * (PCKh[lank] + PCKh[rank])),
            ('Mean', np.sum(PCKh * jnt_ratio)),
            ('Mean@0.1', np.sum(pckAll[11, :] * jnt_ratio))
        ]
        name_value = OrderedDict(name_value)

        return name_value, name_value['Mean']

# --- End of .\lib\dataset\mpii.py ---

# --- Start of .\lib\dataset\__init__.py ---
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .mpii import MPIIDataset as mpii
from .coco import COCODataset as coco

# --- End of .\lib\dataset\__init__.py ---

# --- Start of .\lib\models\pose_hrnet.py ---
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
# our multi-scale bottleneck
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)

        self.planes_chanel = planes // 8
        self.conv2_1 = nn.Conv2d(self.planes_chanel, self.planes_chanel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(self.planes_chanel, momentum=BN_MOMENTUM)
        self.conv2_2 = nn.Conv2d(self.planes_chanel, self.planes_chanel, kernel_size=5, stride=stride, padding=2, bias=False)
        self.bn2_2 = nn.BatchNorm2d(self.planes_chanel, momentum=BN_MOMENTUM)
        self.conv2_3 = nn.Conv2d(self.planes_chanel, self.planes_chanel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2_3 = nn.BatchNorm2d(self.planes_chanel, momentum=BN_MOMENTUM)
        self.conv2_4 = nn.Conv2d(self.planes_chanel, self.planes_chanel, kernel_size=5, stride=stride, padding=2,bias=False)
        self.bn2_4 = nn.BatchNorm2d(self.planes_chanel, momentum=BN_MOMENTUM)
        self.conv2_5 = nn.Conv2d(self.planes_chanel, self.planes_chanel, kernel_size=3, stride=stride, padding=1,bias=False)
        self.bn2_5 = nn.BatchNorm2d(self.planes_chanel, momentum=BN_MOMENTUM)
        self.conv2_6 = nn.Conv2d(self.planes_chanel, self.planes_chanel, kernel_size=5, stride=stride, padding=2,bias=False)
        self.bn2_6 = nn.BatchNorm2d(self.planes_chanel, momentum=BN_MOMENTUM)
        self.conv2_7 = nn.Conv2d(self.planes_chanel, self.planes_chanel, kernel_size=3, stride=stride, padding=1,bias=False)
        self.bn2_7 = nn.BatchNorm2d(self.planes_chanel, momentum=BN_MOMENTUM)
        self.conv2_8 = nn.Conv2d(self.planes_chanel, self.planes_chanel, kernel_size=5, stride=stride, padding=2,bias=False)
        self.bn2_8 = nn.BatchNorm2d(self.planes_chanel, momentum=BN_MOMENTUM)

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(self.bn1(out))

        spx = torch.split(out, self.planes_chanel, 1)

        y1 = self.conv2_1(spx[0])
        y1 = self.relu(self.bn2_1(y1))

        y2 = self.conv2_2(spx[1] + y1)
        y2 = self.relu(self.bn2_2(y2))

        y3 = self.conv2_3(spx[2] + y2)
        y3 = self.relu(self.bn2_3(y3))

        y4 = self.conv2_4(spx[3] + y3)
        y4 = self.relu(self.bn2_4(y4))

        y5 = self.conv2_5(spx[4] + y4)
        y5 = self.relu(self.bn2_5(y5))

        y6 = self.conv2_6(spx[5] + y5)
        y6 = self.relu(self.bn2_6(y6))

        y7 = self.conv2_7(spx[6] + y6)
        y7 = self.relu(self.bn2_7(y7))

        y8 = self.conv2_8(spx[7] + y7)
        y8 = self.relu(self.bn2_8(y8))

        z8 = y8
        z7 = z8 + y7
        z6 = z7 + y6
        z5 = z6 + y5
        z4 = z5 + y4
        z3 = z4 + y3
        z2 = z3 + y2
        z1 = z2 + y1

        z = torch.cat((z1,z2,z3,z4,z5,z6,z7,z8),1)

        out = self.conv3(z)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(
                    num_channels[branch_index] * block.expansion,
                    momentum=BN_MOMENTUM
                ),
            )

        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample
            )
        )
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index]
                )
            )

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels)
            )

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[j],
                                num_inchannels[i],
                                1, 1, 0, bias=False
                            ),
                            nn.BatchNorm2d(num_inchannels[i]),
                            nn.Upsample(scale_factor=2**(j-i), mode='nearest')
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3)
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3),
                                    nn.ReLU(True)
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class PoseHighResolutionNet(nn.Module):

    def __init__(self, cfg, **kwargs):
        self.inplanes = 64
        extra = cfg['MODEL']['EXTRA']
        super(PoseHighResolutionNet, self).__init__()

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 4)

        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=False)

        self.final_layer = nn.Conv2d(
            in_channels=pre_stage_channels[0],
            out_channels=cfg['MODEL']['NUM_JOINTS'],
            kernel_size=extra['FINAL_CONV_KERNEL'],
            stride=1,
            padding=1 if extra['FINAL_CONV_KERNEL'] == 3 else 0
        )

        self.pretrained_layers = extra['PRETRAINED_LAYERS']

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3, 1, 1, bias=False
                            ),
                            nn.BatchNorm2d(num_channels_cur_layer[i]),
                            nn.ReLU(inplace=True)
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(
                                inchannels, outchannels, 3, 2, 1, bias=False
                            ),
                            nn.BatchNorm2d(outchannels),
                            nn.ReLU(inplace=True)
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        x = self.final_layer(y_list[0])

        return x

    def init_weights(self, pretrained=''):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers \
                   or self.pretrained_layers[0] is '*':
                    need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=False)
        elif pretrained:
            logger.error('=> please download pre-trained models first!')
            raise ValueError('{} is not exist!'.format(pretrained))

class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class Bottleneck_cbam(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_cbam, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.ca1 = ChannelAttention(planes)
        self.sa1 = SpatialAttention()

        self.conv4 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm2d(planes)
        self.conv5 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(planes)
        self.conv6 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn6 = nn.BatchNorm2d(planes)
        self.ca2 = ChannelAttention(planes)
        self.sa2 = SpatialAttention()

        self.conv7 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn7 = nn.BatchNorm2d(planes)
        self.conv8 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(planes)
        self.conv9 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn9 = nn.BatchNorm2d(planes)
        self.ca3 = ChannelAttention(planes)
        self.sa3 = SpatialAttention()

        self.conv10 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn10 = nn.BatchNorm2d(planes)
        self.conv11 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn11 = nn.BatchNorm2d(planes)
        self.conv12 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn12 = nn.BatchNorm2d(planes)
        self.ca4 = ChannelAttention(planes)
        self.sa4 = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual1 = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.ca1(out) * out
        out = self.sa1(out) * out
        if self.downsample is not None:
            residual1 = self.downsample(x)
        out += residual1
        out = self.relu(out)

        residual2 = out
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.conv5(out)
        out = self.bn5(out)
        out = self.relu(out)
        out = self.conv6(out)
        out = self.bn6(out)
        out = self.ca2(out) * out
        out = self.sa2(out) * out
        if self.downsample is not None:
            residual2 = self.downsample(out)
        out += residual2
        out = self.relu(out)

        residual3 = out
        out = self.conv7(out)
        out = self.bn7(out)
        out = self.relu(out)
        out = self.conv8(out)
        out = self.bn8(out)
        out = self.relu(out)
        out = self.conv9(out)
        out = self.bn9(out)
        out = self.ca3(out) * out
        out = self.sa3(out) * out
        if self.downsample is not None:
            residual3 = self.downsample(out)
        out += residual3
        out = self.relu(out)

        residual4 = out
        out = self.conv10(out)
        out = self.bn10(out)
        out = self.relu(out)
        out = self.conv11(out)
        out = self.bn11(out)
        out = self.relu(out)
        out = self.conv12(out)
        out = self.bn12(out)
        out = self.ca4(out) * out
        out = self.sa4(out) * out
        if self.downsample is not None:
            residual4 = self.downsample(out)
        out += residual4
        out = self.relu(out)

        return out

def get_pose_net(cfg, is_train, **kwargs):
    model = PoseHighResolutionNet(cfg, **kwargs)
    model_fine = Bottleneck_cbam(cfg.MODEL.NUM_JOINTS,cfg.MODEL.NUM_JOINTS)
    if is_train and cfg['MODEL']['INIT_WEIGHTS']:
        model.init_weights(cfg['MODEL']['PRETRAINED'])

    return model, model_fine
# --- End of .\lib\models\pose_hrnet.py ---

# --- Start of .\lib\models\pose_resnet.py ---
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class PoseResNet(nn.Module):

    def __init__(self, block, layers, cfg, **kwargs):
        self.inplanes = 64
        extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS

        super(PoseResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            extra.NUM_DECONV_LAYERS,
            extra.NUM_DECONV_FILTERS,
            extra.NUM_DECONV_KERNELS,
        )

        self.final_layer = nn.Conv2d(
            in_channels=extra.NUM_DECONV_FILTERS[-1],
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.deconv_layers(x)
        x = self.final_layer(x)

        return x

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            logger.info('=> init deconv weights from normal distribution')
            for name, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    logger.info('=> init {}.weight as 1'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            logger.info('=> init final conv weights from normal distribution')
            for m in self.final_layer.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)

            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            self.load_state_dict(pretrained_state_dict, strict=False)
        else:
            logger.info('=> init weights from normal distribution')
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.normal_(m.weight, std=0.001)
                    # nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)


resnet_spec = {
    18: (BasicBlock, [2, 2, 2, 2]),
    34: (BasicBlock, [3, 4, 6, 3]),
    50: (Bottleneck, [3, 4, 6, 3]),
    101: (Bottleneck, [3, 4, 23, 3]),
    152: (Bottleneck, [3, 8, 36, 3])
}


def get_pose_net(cfg, is_train, **kwargs):
    num_layers = cfg.MODEL.EXTRA.NUM_LAYERS

    block_class, layers = resnet_spec[num_layers]

    model = PoseResNet(block_class, layers, cfg, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)

    return model

# --- End of .\lib\models\pose_resnet.py ---

# --- Start of .\lib\models\__init__.py ---
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import models.pose_resnet
import models.pose_hrnet

# --- End of .\lib\models\__init__.py ---

# --- Start of .\lib\nms\nms.py ---
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Modified from py-faster-rcnn (https://github.com/rbgirshick/py-faster-rcnn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from .cpu_nms import cpu_nms
from .gpu_nms import gpu_nms


def py_nms_wrapper(thresh):
    def _nms(dets):
        return nms(dets, thresh)
    return _nms


def cpu_nms_wrapper(thresh):
    def _nms(dets):
        return cpu_nms(dets, thresh)
    return _nms


def gpu_nms_wrapper(thresh, device_id):
    def _nms(dets):
        return gpu_nms(dets, thresh, device_id)
    return _nms


def nms(dets, thresh):
    """
    greedily select boxes with high confidence and overlap with current maximum <= thresh
    rule out overlap >= thresh
    :param dets: [[x1, y1, x2, y2 score]]
    :param thresh: retain overlap < thresh
    :return: indexes to keep
    """
    if dets.shape[0] == 0:
        return []

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def oks_iou(g, d, a_g, a_d, sigmas=None, in_vis_thre=None):
    if not isinstance(sigmas, np.ndarray):
        sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0
    vars = (sigmas * 2) ** 2
    xg = g[0::3]
    yg = g[1::3]
    vg = g[2::3]
    ious = np.zeros((d.shape[0]))
    for n_d in range(0, d.shape[0]):
        xd = d[n_d, 0::3]
        yd = d[n_d, 1::3]
        vd = d[n_d, 2::3]
        dx = xd - xg
        dy = yd - yg
        e = (dx ** 2 + dy ** 2) / vars / ((a_g + a_d[n_d]) / 2 + np.spacing(1)) / 2
        if in_vis_thre is not None:
            ind = list(vg > in_vis_thre) and list(vd > in_vis_thre)
            e = e[ind]
        ious[n_d] = np.sum(np.exp(-e)) / e.shape[0] if e.shape[0] != 0 else 0.0
    return ious


def oks_nms(kpts_db, thresh, sigmas=None, in_vis_thre=None):
    """
    greedily select boxes with high confidence and overlap with current maximum <= thresh
    rule out overlap >= thresh, overlap = oks
    :param kpts_db
    :param thresh: retain overlap < thresh
    :return: indexes to keep
    """
    if len(kpts_db) == 0:
        return []

    scores = np.array([kpts_db[i]['score'] for i in range(len(kpts_db))])
    kpts = np.array([kpts_db[i]['keypoints'].flatten() for i in range(len(kpts_db))])
    areas = np.array([kpts_db[i]['area'] for i in range(len(kpts_db))])

    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        oks_ovr = oks_iou(kpts[i], kpts[order[1:]], areas[i], areas[order[1:]], sigmas, in_vis_thre)

        inds = np.where(oks_ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def rescore(overlap, scores, thresh, type='gaussian'):
    assert overlap.shape[0] == scores.shape[0]
    if type == 'linear':
        inds = np.where(overlap >= thresh)[0]
        scores[inds] = scores[inds] * (1 - overlap[inds])
    else:
        scores = scores * np.exp(- overlap**2 / thresh)

    return scores


def soft_oks_nms(kpts_db, thresh, sigmas=None, in_vis_thre=None):
    """
    greedily select boxes with high confidence and overlap with current maximum <= thresh
    rule out overlap >= thresh, overlap = oks
    :param kpts_db
    :param thresh: retain overlap < thresh
    :return: indexes to keep
    """
    if len(kpts_db) == 0:
        return []

    scores = np.array([kpts_db[i]['score'] for i in range(len(kpts_db))])
    kpts = np.array([kpts_db[i]['keypoints'].flatten() for i in range(len(kpts_db))])
    areas = np.array([kpts_db[i]['area'] for i in range(len(kpts_db))])

    order = scores.argsort()[::-1]
    scores = scores[order]

    # max_dets = order.size
    max_dets = 20
    keep = np.zeros(max_dets, dtype=np.intp)
    keep_cnt = 0
    while order.size > 0 and keep_cnt < max_dets:
        i = order[0]

        oks_ovr = oks_iou(kpts[i], kpts[order[1:]], areas[i], areas[order[1:]], sigmas, in_vis_thre)

        order = order[1:]
        scores = rescore(oks_ovr, scores[1:], thresh)

        tmp = scores.argsort()[::-1]
        order = order[tmp]
        scores = scores[tmp]

        keep[keep_cnt] = i
        keep_cnt += 1

    keep = keep[:keep_cnt]

    return keep
    # kpts_db = kpts_db[:keep_cnt]

    # return kpts_db

# --- End of .\lib\nms\nms.py ---

# --- Start of .\lib\nms\setup_linux.py ---
# --------------------------------------------------------
# Pose.gluon
# Copyright (c) 2018-present Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified from py-faster-rcnn (https://github.com/rbgirshick/py-faster-rcnn)
# --------------------------------------------------------

import os
from os.path import join as pjoin
from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np


def find_in_path(name, path):
    "Find a file in a search path"
    # Adapted fom
    # http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_cuda():
    """Locate the CUDA environment on the system
    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.
    Starts by looking for the CUDAHOME env variable. If not found, everything
    is based on finding 'nvcc' in the PATH.
    """

    # first check if the CUDAHOME env variable is in use
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = pjoin(home, 'bin', 'nvcc')
    else:
        # otherwise, search the PATH for NVCC
        default_path = pjoin(os.sep, 'usr', 'local', 'cuda', 'bin')
        nvcc = find_in_path('nvcc', os.environ['PATH'] + os.pathsep + default_path)
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be '
                'located in your $PATH. Either add it to your path, or set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home':home, 'nvcc':nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, 'lib64')}
    for k, v in cudaconfig.items():
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be located in %s' % (k, v))

    return cudaconfig
CUDA = locate_cuda()


# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()


def customize_compiler_for_nvcc(self):
    """inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.
    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on."""

    # tell the compiler it can processes .cu
    self.src_extensions.append('.cu')

    # save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu':
            # use the cuda for .cu files
            self.set_executable('compiler_so', CUDA['nvcc'])
            # use only a subset of the extra_postargs, which are 1-1 translated
            # from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    self._compile = _compile


# run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)


ext_modules = [
    Extension(
        "cpu_nms",
        ["cpu_nms.pyx"],
        extra_compile_args={'gcc': ["-Wno-cpp", "-Wno-unused-function"]},
        include_dirs = [numpy_include]
    ),
    Extension('gpu_nms',
        ['nms_kernel.cu', 'gpu_nms.pyx'],
        library_dirs=[CUDA['lib64']],
        libraries=['cudart'],
        language='c++',
        runtime_library_dirs=[CUDA['lib64']],
        # this syntax is specific to this build system
        # we're only going to use certain compiler args with nvcc and not with
        # gcc the implementation of this trick is in customize_compiler() below
        extra_compile_args={'gcc': ["-Wno-unused-function"],
                            'nvcc': ['-arch=sm_70',
                                     '--ptxas-options=-v',
                                     '-c',
                                     '--compiler-options',
                                     "'-fPIC'"]},
        include_dirs = [numpy_include, CUDA['include']]
    ),
]

setup(
    name='nms',
    ext_modules=ext_modules,
    # inject our custom trigger
    cmdclass={'build_ext': custom_build_ext},
)

# --- End of .\lib\nms\setup_linux.py ---

# --- Start of .\lib\nms\__init__.py ---

# --- End of .\lib\nms\__init__.py ---

# --- Start of .\lib\utils\transforms.py ---
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2


def flip_back(output_flipped, matched_parts):
    '''
    ouput_flipped: numpy.ndarray(batch_size, num_joints, height, width)
    '''
    assert output_flipped.ndim == 4,\
        'output_flipped should be [batch_size, num_joints, height, width]'

    output_flipped = output_flipped[:, :, :, ::-1]

    for pair in matched_parts:
        tmp = output_flipped[:, pair[0], :, :].copy()
        output_flipped[:, pair[0], :, :] = output_flipped[:, pair[1], :, :]
        output_flipped[:, pair[1], :, :] = tmp

    return output_flipped


def fliplr_joints(joints, joints_vis, width, matched_parts):
    """
    flip coords
    """
    # Flip horizontal
    joints[:, 0] = width - joints[:, 0] - 1

    # Change left-right parts
    for pair in matched_parts:
        joints[pair[0], :], joints[pair[1], :] = \
            joints[pair[1], :], joints[pair[0], :].copy()
        joints_vis[pair[0], :], joints_vis[pair[1], :] = \
            joints_vis[pair[1], :], joints_vis[pair[0], :].copy()

    return joints*joints_vis, joints_vis


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def get_affine_transform(
        center, scale, rot, output_size,
        shift=np.array([0, 0], dtype=np.float32), inv=0
):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def crop(img, center, scale, output_size, rot=0):
    trans = get_affine_transform(center, scale, rot, output_size)

    dst_img = cv2.warpAffine(
        img, trans, (int(output_size[0]), int(output_size[1])),
        flags=cv2.INTER_LINEAR
    )

    return dst_img

# --- End of .\lib\utils\transforms.py ---

# --- Start of .\lib\utils\utils.py ---
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from collections import namedtuple
from pathlib import Path

import torch
import torch.optim as optim
import torch.nn as nn


def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.DATASET + '_' + cfg.DATASET.HYBRID_JOINTS_TYPE \
        if cfg.DATASET.HYBRID_JOINTS_TYPE else cfg.DATASET.DATASET
    dataset = dataset.replace(':', '_')
    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / dataset / model / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
        (cfg_name + '_' + time_str)

    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)


def get_optimizer(cfg, model):
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.TRAIN.LR
        )

    return optimizer


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best and 'state_dict' in states:
        torch.save(states['best_state_dict'],
                   os.path.join(output_dir, 'model_best.pth'))


def get_model_summary(model, *input_tensors, item_length=26, verbose=False):
    """
    :param model:
    :param input_tensors:
    :param item_length:
    :return:
    """

    summary = []

    ModuleDetails = namedtuple(
        "Layer", ["name", "input_size", "output_size", "num_parameters", "multiply_adds"])
    hooks = []
    layer_instances = {}

    def add_hooks(module):

        def hook(module, input, output):
            class_name = str(module.__class__.__name__)

            instance_index = 1
            if class_name not in layer_instances:
                layer_instances[class_name] = instance_index
            else:
                instance_index = layer_instances[class_name] + 1
                layer_instances[class_name] = instance_index

            layer_name = class_name + "_" + str(instance_index)

            params = 0

            if class_name.find("Conv") != -1 or class_name.find("BatchNorm") != -1 or \
               class_name.find("Linear") != -1:
                for param_ in module.parameters():
                    params += param_.view(-1).size(0)

            flops = "Not Available"
            if class_name.find("Conv") != -1 and hasattr(module, "weight"):
                flops = (
                    torch.prod(
                        torch.LongTensor(list(module.weight.data.size()))) *
                    torch.prod(
                        torch.LongTensor(list(output.size())[2:]))).item()
            elif isinstance(module, nn.Linear):
                flops = (torch.prod(torch.LongTensor(list(output.size()))) \
                         * input[0].size(1)).item()

            if isinstance(input[0], list):
                input = input[0]
            if isinstance(output, list):
                output = output[0]

            summary.append(
                ModuleDetails(
                    name=layer_name,
                    input_size=list(input[0].size()),
                    output_size=list(output.size()),
                    num_parameters=params,
                    multiply_adds=flops)
            )

        if not isinstance(module, nn.ModuleList) \
           and not isinstance(module, nn.Sequential) \
           and module != model:
            hooks.append(module.register_forward_hook(hook))

    model.eval()
    model.apply(add_hooks)

    space_len = item_length

    model(*input_tensors)
    for hook in hooks:
        hook.remove()

    details = ''
    if verbose:
        details = "Model Summary" + \
            os.linesep + \
            "Name{}Input Size{}Output Size{}Parameters{}Multiply Adds (Flops){}".format(
                ' ' * (space_len - len("Name")),
                ' ' * (space_len - len("Input Size")),
                ' ' * (space_len - len("Output Size")),
                ' ' * (space_len - len("Parameters")),
                ' ' * (space_len - len("Multiply Adds (Flops)"))) \
                + os.linesep + '-' * space_len * 5 + os.linesep

    params_sum = 0
    flops_sum = 0
    for layer in summary:
        params_sum += layer.num_parameters
        if layer.multiply_adds != "Not Available":
            flops_sum += layer.multiply_adds
        if verbose:
            details += "{}{}{}{}{}{}{}{}{}{}".format(
                layer.name,
                ' ' * (space_len - len(layer.name)),
                layer.input_size,
                ' ' * (space_len - len(str(layer.input_size))),
                layer.output_size,
                ' ' * (space_len - len(str(layer.output_size))),
                layer.num_parameters,
                ' ' * (space_len - len(str(layer.num_parameters))),
                layer.multiply_adds,
                ' ' * (space_len - len(str(layer.multiply_adds)))) \
                + os.linesep + '-' * space_len * 5 + os.linesep

    details += os.linesep \
        + "Total Parameters: {:,}".format(params_sum) \
        + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Total Multiply Adds (For Convolution and Linear Layers only): {:,} GFLOPs".format(flops_sum/(1024**3)) \
        + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Number of Layers" + os.linesep
    for layer in layer_instances:
        details += "{} : {} layers   ".format(layer, layer_instances[layer])

    return details

# --- End of .\lib\utils\utils.py ---

# --- Start of .\lib\utils\vis.py ---
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import torchvision
import cv2

from core.inference import get_max_preds


def save_batch_image_with_joints(batch_image, batch_joints, batch_joints_vis,
                                 file_name, nrow=8, padding=2):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            joints = batch_joints[k]
            joints_vis = batch_joints_vis[k]

            for joint, joint_vis in zip(joints, joints_vis):
                joint[0] = x * width + padding + joint[0]
                joint[1] = y * height + padding + joint[1]
                if joint_vis[0]:
                    cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 2, [255, 0, 0], 2)
            k = k + 1
    cv2.imwrite(file_name, ndarr)


def save_batch_heatmaps(batch_image, batch_heatmaps, file_name,
                        normalize=True):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    file_name: saved file name
    '''
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    heatmap_height = batch_heatmaps.size(2)
    heatmap_width = batch_heatmaps.size(3)

    grid_image = np.zeros((batch_size*heatmap_height,
                           (num_joints+1)*heatmap_width,
                           3),
                          dtype=np.uint8)

    preds, maxvals = get_max_preds(batch_heatmaps.detach().cpu().numpy())

    for i in range(batch_size):
        image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .byte()\
                              .permute(1, 2, 0)\
                              .cpu().numpy()
        heatmaps = batch_heatmaps[i].mul(255)\
                                    .clamp(0, 255)\
                                    .byte()\
                                    .cpu().numpy()

        resized_image = cv2.resize(image,
                                   (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_joints):
            cv2.circle(resized_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)
            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image = colored_heatmap*0.7 + resized_image*0.3
            cv2.circle(masked_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)

            width_begin = heatmap_width * (j+1)
            width_end = heatmap_width * (j+2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image
            # grid_image[height_begin:height_end, width_begin:width_end, :] = \
            #     colored_heatmap*0.7 + resized_image*0.3

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image

    cv2.imwrite(file_name, grid_image)


def save_debug_images(config, input, meta, target, joints_pred, output,
                      prefix):
    if not config.DEBUG.DEBUG:
        return

    if config.DEBUG.SAVE_BATCH_IMAGES_GT:
        save_batch_image_with_joints(
            input, meta['joints'], meta['joints_vis'],
            '{}_gt.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_BATCH_IMAGES_PRED:
        save_batch_image_with_joints(
            input, joints_pred, meta['joints_vis'],
            '{}_pred.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_HEATMAPS_GT:
        save_batch_heatmaps(
            input, target, '{}_hm_gt.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_HEATMAPS_PRED:
        save_batch_heatmaps(
            input, output, '{}_hm_pred.jpg'.format(prefix)
        )

# --- End of .\lib\utils\vis.py ---

# --- Start of .\lib\utils\zipreader.py ---
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import zipfile
import xml.etree.ElementTree as ET

import cv2
import numpy as np

_im_zfile = []
_xml_path_zip = []
_xml_zfile = []


def imread(filename, flags=cv2.IMREAD_COLOR):
    global _im_zfile
    path = filename
    pos_at = path.index('@')
    if pos_at == -1:
        print("character '@' is not found from the given path '%s'"%(path))
        assert 0
    path_zip = path[0: pos_at]
    path_img = path[pos_at + 2:]
    if not os.path.isfile(path_zip):
        print("zip file '%s' is not found"%(path_zip))
        assert 0
    for i in range(len(_im_zfile)):
        if _im_zfile[i]['path'] == path_zip:
            data = _im_zfile[i]['zipfile'].read(path_img)
            return cv2.imdecode(np.frombuffer(data, np.uint8), flags)

    _im_zfile.append({
        'path': path_zip,
        'zipfile': zipfile.ZipFile(path_zip, 'r')
    })
    data = _im_zfile[-1]['zipfile'].read(path_img)

    return cv2.imdecode(np.frombuffer(data, np.uint8), flags)


def xmlread(filename):
    global _xml_path_zip
    global _xml_zfile
    path = filename
    pos_at = path.index('@')
    if pos_at == -1:
        print("character '@' is not found from the given path '%s'"%(path))
        assert 0
    path_zip = path[0: pos_at]
    path_xml = path[pos_at + 2:]
    if not os.path.isfile(path_zip):
        print("zip file '%s' is not found"%(path_zip))
        assert 0
    for i in xrange(len(_xml_path_zip)):
        if _xml_path_zip[i] == path_zip:
            data = _xml_zfile[i].open(path_xml)
            return ET.fromstring(data.read())
    _xml_path_zip.append(path_zip)
    print("read new xml file '%s'"%(path_zip))
    _xml_zfile.append(zipfile.ZipFile(path_zip, 'r'))
    data = _xml_zfile[-1].open(path_xml)
    return ET.fromstring(data.read())

# --- End of .\lib\utils\zipreader.py ---

# --- Start of .\lib\utils\__init__.py ---

# --- End of .\lib\utils\__init__.py ---

# --- Start of .\output\coco\pose_hrnet\w48_384x288_adam_lr1e-3\pose_hrnet.py ---
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
# our multi-scale bottleneck
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)

        self.planes_chanel = planes // 8
        self.conv2_1 = nn.Conv2d(self.planes_chanel, self.planes_chanel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(self.planes_chanel, momentum=BN_MOMENTUM)
        self.conv2_2 = nn.Conv2d(self.planes_chanel, self.planes_chanel, kernel_size=5, stride=stride, padding=2, bias=False)
        self.bn2_2 = nn.BatchNorm2d(self.planes_chanel, momentum=BN_MOMENTUM)
        self.conv2_3 = nn.Conv2d(self.planes_chanel, self.planes_chanel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2_3 = nn.BatchNorm2d(self.planes_chanel, momentum=BN_MOMENTUM)
        self.conv2_4 = nn.Conv2d(self.planes_chanel, self.planes_chanel, kernel_size=5, stride=stride, padding=2,bias=False)
        self.bn2_4 = nn.BatchNorm2d(self.planes_chanel, momentum=BN_MOMENTUM)
        self.conv2_5 = nn.Conv2d(self.planes_chanel, self.planes_chanel, kernel_size=3, stride=stride, padding=1,bias=False)
        self.bn2_5 = nn.BatchNorm2d(self.planes_chanel, momentum=BN_MOMENTUM)
        self.conv2_6 = nn.Conv2d(self.planes_chanel, self.planes_chanel, kernel_size=5, stride=stride, padding=2,bias=False)
        self.bn2_6 = nn.BatchNorm2d(self.planes_chanel, momentum=BN_MOMENTUM)
        self.conv2_7 = nn.Conv2d(self.planes_chanel, self.planes_chanel, kernel_size=3, stride=stride, padding=1,bias=False)
        self.bn2_7 = nn.BatchNorm2d(self.planes_chanel, momentum=BN_MOMENTUM)
        self.conv2_8 = nn.Conv2d(self.planes_chanel, self.planes_chanel, kernel_size=5, stride=stride, padding=2,bias=False)
        self.bn2_8 = nn.BatchNorm2d(self.planes_chanel, momentum=BN_MOMENTUM)

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(self.bn1(out))

        spx = torch.split(out, self.planes_chanel, 1)

        y1 = self.conv2_1(spx[0])
        y1 = self.relu(self.bn2_1(y1))

        y2 = self.conv2_2(spx[1] + y1)
        y2 = self.relu(self.bn2_2(y2))

        y3 = self.conv2_3(spx[2] + y2)
        y3 = self.relu(self.bn2_3(y3))

        y4 = self.conv2_4(spx[3] + y3)
        y4 = self.relu(self.bn2_4(y4))

        y5 = self.conv2_5(spx[4] + y4)
        y5 = self.relu(self.bn2_5(y5))

        y6 = self.conv2_6(spx[5] + y5)
        y6 = self.relu(self.bn2_6(y6))

        y7 = self.conv2_7(spx[6] + y6)
        y7 = self.relu(self.bn2_7(y7))

        y8 = self.conv2_8(spx[7] + y7)
        y8 = self.relu(self.bn2_8(y8))

        z8 = y8
        z7 = z8 + y7
        z6 = z7 + y6
        z5 = z6 + y5
        z4 = z5 + y4
        z3 = z4 + y3
        z2 = z3 + y2
        z1 = z2 + y1

        z = torch.cat((z1,z2,z3,z4,z5,z6,z7,z8),1)

        out = self.conv3(z)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(
                    num_channels[branch_index] * block.expansion,
                    momentum=BN_MOMENTUM
                ),
            )

        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample
            )
        )
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index]
                )
            )

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels)
            )

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[j],
                                num_inchannels[i],
                                1, 1, 0, bias=False
                            ),
                            nn.BatchNorm2d(num_inchannels[i]),
                            nn.Upsample(scale_factor=2**(j-i), mode='nearest')
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3)
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3),
                                    nn.ReLU(True)
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class PoseHighResolutionNet(nn.Module):

    def __init__(self, cfg, **kwargs):
        self.inplanes = 64
        extra = cfg['MODEL']['EXTRA']
        super(PoseHighResolutionNet, self).__init__()

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 4)

        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=False)

        self.final_layer = nn.Conv2d(
            in_channels=pre_stage_channels[0],
            out_channels=cfg['MODEL']['NUM_JOINTS'],
            kernel_size=extra['FINAL_CONV_KERNEL'],
            stride=1,
            padding=1 if extra['FINAL_CONV_KERNEL'] == 3 else 0
        )

        self.pretrained_layers = extra['PRETRAINED_LAYERS']

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3, 1, 1, bias=False
                            ),
                            nn.BatchNorm2d(num_channels_cur_layer[i]),
                            nn.ReLU(inplace=True)
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(
                                inchannels, outchannels, 3, 2, 1, bias=False
                            ),
                            nn.BatchNorm2d(outchannels),
                            nn.ReLU(inplace=True)
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        x = self.final_layer(y_list[0])

        return x

    def init_weights(self, pretrained=''):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers \
                   or self.pretrained_layers[0] is '*':
                    need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=False)
        elif pretrained:
            logger.error('=> please download pre-trained models first!')
            raise ValueError('{} is not exist!'.format(pretrained))

class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class Bottleneck_cbam(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_cbam, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.ca1 = ChannelAttention(planes)
        self.sa1 = SpatialAttention()

        self.conv4 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm2d(planes)
        self.conv5 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(planes)
        self.conv6 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn6 = nn.BatchNorm2d(planes)
        self.ca2 = ChannelAttention(planes)
        self.sa2 = SpatialAttention()

        self.conv7 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn7 = nn.BatchNorm2d(planes)
        self.conv8 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(planes)
        self.conv9 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn9 = nn.BatchNorm2d(planes)
        self.ca3 = ChannelAttention(planes)
        self.sa3 = SpatialAttention()

        self.conv10 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn10 = nn.BatchNorm2d(planes)
        self.conv11 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn11 = nn.BatchNorm2d(planes)
        self.conv12 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn12 = nn.BatchNorm2d(planes)
        self.ca4 = ChannelAttention(planes)
        self.sa4 = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual1 = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.ca1(out) * out
        out = self.sa1(out) * out
        if self.downsample is not None:
            residual1 = self.downsample(x)
        out += residual1
        out = self.relu(out)

        residual2 = out
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.conv5(out)
        out = self.bn5(out)
        out = self.relu(out)
        out = self.conv6(out)
        out = self.bn6(out)
        out = self.ca2(out) * out
        out = self.sa2(out) * out
        if self.downsample is not None:
            residual2 = self.downsample(out)
        out += residual2
        out = self.relu(out)

        residual3 = out
        out = self.conv7(out)
        out = self.bn7(out)
        out = self.relu(out)
        out = self.conv8(out)
        out = self.bn8(out)
        out = self.relu(out)
        out = self.conv9(out)
        out = self.bn9(out)
        out = self.ca3(out) * out
        out = self.sa3(out) * out
        if self.downsample is not None:
            residual3 = self.downsample(out)
        out += residual3
        out = self.relu(out)

        residual4 = out
        out = self.conv10(out)
        out = self.bn10(out)
        out = self.relu(out)
        out = self.conv11(out)
        out = self.bn11(out)
        out = self.relu(out)
        out = self.conv12(out)
        out = self.bn12(out)
        out = self.ca4(out) * out
        out = self.sa4(out) * out
        if self.downsample is not None:
            residual4 = self.downsample(out)
        out += residual4
        out = self.relu(out)

        return out

def get_pose_net(cfg, is_train, **kwargs):
    model = PoseHighResolutionNet(cfg, **kwargs)
    model_fine = Bottleneck_cbam(cfg.MODEL.NUM_JOINTS,cfg.MODEL.NUM_JOINTS)
    if is_train and cfg['MODEL']['INIT_WEIGHTS']:
        model.init_weights(cfg['MODEL']['PRETRAINED'])

    return model, model_fine
# --- End of .\output\coco\pose_hrnet\w48_384x288_adam_lr1e-3\pose_hrnet.py ---

# --- Start of .\tools\test.py ---
# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.loss import JointsOHKMMSELoss
from core.function import validate
from utils.utils import create_logger

import dataset
import models


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model, model_fine = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(
        cfg, is_train=False)

    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        model_state_file = os.path.join(
            final_output_dir, 'final_state.pth'
        )
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
    model_fine = torch.nn.DataParallel(model_fine, device_ids=[0]).cuda()

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()
    criterion_fine = JointsOHKMMSELoss(False).cuda()

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    valid_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True
    )

    # evaluate on validation set
    validate(
        cfg, valid_loader, valid_dataset, model, model_fine, criterion, criterion_fine,
        final_output_dir, tb_log_dir)


if __name__ == '__main__':
    main()

# --- End of .\tools\test.py ---

# --- Start of .\tools\train.py ---
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import _init_paths
from config import cfg
from config import update_config

from core.loss import JointsMSELoss
from core.loss import JointsOHKMMSELoss
from core.function import train
from core.function import validate
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.utils import get_model_summary

import dataset
import models


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model, model_fine = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=True)

    # copy model file
    this_dir = os.path.dirname(__file__)
    shutil.copy2(
        os.path.join(this_dir, '../lib/models', cfg.MODEL.NAME + '.py'),
        final_output_dir)

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    dump_input = torch.rand(
        (1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0])
    )
    writer_dict['writer'].add_graph(model, (dump_input, ))
    logger.info(get_model_summary(model, dump_input))

    dump_input_fine = torch.rand(1, cfg.MODEL.NUM_JOINTS, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0])  #gaidong
    writer_dict['writer'].add_graph(model_fine, (dump_input_fine, ))
    logger.info(get_model_summary(model_fine, dump_input_fine))

    #model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
    #model_fine = torch.nn.DataParallel(model_fine,device_ids=[0]).cuda()         # gaidong
    model = torch.nn.DataParallel(model, device_ids=[0,1,2,3]).cuda()
    model_fine = torch.nn.DataParallel(model_fine, device_ids=[0,1,2,3]).cuda()         # gaidong
    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()
    criterion_fine = JointsOHKMMSELoss(False).cuda()
    # Data loading code
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
    valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    best_perf = 0.0
    best_model = False
    last_epoch = -1
    optimizer=torch.optim.Adam([{"params":model.parameters()}, {"params":model_fine.parameters()}], lr=cfg.TRAIN.LR) #gaidong
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth'
    )

    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=last_epoch
    )

    # 訓練開始前釋放記憶體
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        # train for one epoch
        train(cfg, train_loader, model, model_fine, criterion, criterion_fine,optimizer, epoch,
              final_output_dir, tb_log_dir, writer_dict)
        
        # 更新學習率（應該在 optimizer.step() 之後）
        lr_scheduler.step()

        print('lr: ', lr_scheduler.get_last_lr())  # 修正 get_lr() 問題

        # evaluate on validation set
        perf_indicator = validate(
            cfg, valid_loader, valid_dataset, model, model_fine,criterion, criterion_fine,
            final_output_dir, tb_log_dir, writer_dict
        )
        if perf_indicator >= best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': cfg.MODEL.NAME,
            'state_dict': model.state_dict(),
            'best_state_dict': model.module.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir)

    final_model_state_file = os.path.join(
        final_output_dir, 'final_state.pth'
    )
    logger.info('=> saving final model state to {}'.format(
        final_model_state_file)
    )
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()

# --- End of .\tools\train.py ---

# --- Start of .\tools\_init_paths.py ---
# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = osp.dirname(__file__)

lib_path = osp.join(this_dir, '..', 'lib')

add_path(lib_path)

mm_path = osp.join(this_dir, '..', 'lib/poseeval/py-motmetrics')
add_path(mm_path)

if __name__ == '__main__':
    print("sys.path",sys.path)
    print("this_dir",this_dir)
    print("lib_path",lib_path)
# --- End of .\tools\_init_paths.py ---

# --- Start of .\visualization\plot_coco.py ---
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# Modified by Depu Meng (mdp@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os


class ColorStyle:
    def __init__(self, color, link_pairs, point_color):
        self.color = color
        self.link_pairs = link_pairs
        self.point_color = point_color

        for i in range(len(self.color)):
            self.link_pairs[i].append(tuple(np.array(self.color[i]) / 255.))

        self.ring_color = []
        for i in range(len(self.point_color)):
            self.ring_color.append(tuple(np.array(self.point_color[i]) / 255.))


# Xiaochu Style
# (R,G,B)
color1 = [(179, 0, 0), (228, 26, 28), (255, 255, 51),
          (49, 163, 84), (0, 109, 45), (255, 255, 51),
          (240, 2, 127), (240, 2, 127), (240, 2, 127), (240, 2, 127), (240, 2, 127),
          (217, 95, 14), (254, 153, 41), (255, 255, 51),
          (44, 127, 184), (0, 0, 255)]

link_pairs1 = [
    [15, 13], [13, 11], [11, 5],
    [12, 14], [14, 16], [12, 6],
    [3, 1], [1, 2], [1, 0], [0, 2], [2, 4],
    [9, 7], [7, 5], [5, 6],
    [6, 8], [8, 10],
]

point_color1 = [(240, 2, 127), (240, 2, 127), (240, 2, 127),
                (240, 2, 127), (240, 2, 127),
                (255, 255, 51), (255, 255, 51),
                (254, 153, 41), (44, 127, 184),
                (217, 95, 14), (0, 0, 255),
                (255, 255, 51), (255, 255, 51), (228, 26, 28),
                (49, 163, 84), (252, 176, 243), (0, 176, 240),
                (255, 255, 0), (169, 209, 142),
                (255, 255, 0), (169, 209, 142),
                (255, 255, 0), (169, 209, 142)]

xiaochu_style = ColorStyle(color1, link_pairs1, point_color1)

# Chunhua Style
# (R,G,B)
color2 = [(252, 176, 243), (252, 176, 243), (252, 176, 243),
          (0, 176, 240), (0, 176, 240), (0, 176, 240),
          (240, 2, 127), (240, 2, 127), (240, 2, 127), (240, 2, 127), (240, 2, 127),
          (255, 255, 0), (255, 255, 0), (169, 209, 142),
          (169, 209, 142), (169, 209, 142)]

link_pairs2 = [
    [15, 13], [13, 11], [11, 5],
    [12, 14], [14, 16], [12, 6],
    [3, 1], [1, 2], [1, 0], [0, 2], [2, 4],
    [9, 7], [7, 5], [5, 6], [6, 8], [8, 10],
]

point_color2 = [(240, 2, 127), (240, 2, 127), (240, 2, 127),
                (240, 2, 127), (240, 2, 127),
                (255, 255, 0), (169, 209, 142),
                (255, 255, 0), (169, 209, 142),
                (255, 255, 0), (169, 209, 142),
                (252, 176, 243), (0, 176, 240), (252, 176, 243),
                (0, 176, 240), (252, 176, 243), (0, 176, 240),
                (255, 255, 0), (169, 209, 142),
                (255, 255, 0), (169, 209, 142),
                (255, 255, 0), (169, 209, 142)]

chunhua_style = ColorStyle(color2, link_pairs2, point_color2)


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize COCO predictions')
    # general
    parser.add_argument('--image-path',
                        help='Path of COCO val images',
                        type=str,
                        default='data/coco/images/val2017/'
                        )

    parser.add_argument('--gt-anno',
                        help='Path of COCO val annotation',
                        type=str,
                        default='data/coco/annotations/person_keypoints_val2017.json'
                        )

    parser.add_argument('--save-path',
                        help="Path to save the visualizations",
                        type=str,
                        default='visualization/coco/')

    parser.add_argument('--prediction',
                        help="Prediction file to visualize",
                        type=str,
                        required=True)

    parser.add_argument('--style',
                        help="Style of the visualization: Chunhua style or Xiaochu style",
                        type=str,
                        default='chunhua')

    args = parser.parse_args()

    return args


def map_joint_dict(joints):
    joints_dict = {}
    for i in range(joints.shape[0]):
        x = int(joints[i][0])
        y = int(joints[i][1])
        id = i
        joints_dict[id] = (x, y)

    return joints_dict


def plot(data, gt_file, img_path, save_path,
         link_pairs, ring_color, save=True):
    # joints
    coco = COCO(gt_file)
    coco_dt = coco.loadRes(data)
    coco_eval = COCOeval(coco, coco_dt, 'keypoints')
    coco_eval._prepare()
    gts_ = coco_eval._gts
    dts_ = coco_eval._dts

    p = coco_eval.params
    p.imgIds = list(np.unique(p.imgIds))
    if p.useCats:
        p.catIds = list(np.unique(p.catIds))
    p.maxDets = sorted(p.maxDets)

    # loop through images, area range, max detection number
    catIds = p.catIds if p.useCats else [-1]
    threshold = 0.3
    joint_thres = 0.2
    for catId in catIds:
        for imgId in p.imgIds[:5000]:
            # dimention here should be Nxm
            gts = gts_[imgId, catId]
            dts = dts_[imgId, catId]
            inds = np.argsort([-d['score'] for d in dts], kind='mergesort')
            dts = [dts[i] for i in inds]
            if len(dts) > p.maxDets[-1]:
                dts = dts[0:p.maxDets[-1]]
            if len(gts) == 0 or len(dts) == 0:
                continue

            sum_score = 0
            num_box = 0
            img_name = str(imgId).zfill(12)

            # Read Images
            img_file = img_path + img_name + '.jpg'
            data_numpy = cv2.imread(img_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            h = data_numpy.shape[0]
            w = data_numpy.shape[1]

            # Plot
            fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
            ax = plt.subplot(1, 1, 1)
            bk = plt.imshow(data_numpy[:, :, ::-1])
            bk.set_zorder(-1)
            # print(img_name)
            for j, gt in enumerate(gts):
                # matching dt_box and gt_box
                bb = gt['bbox']
                x0 = bb[0] - bb[2];
                x1 = bb[0] + bb[2] * 2
                y0 = bb[1] - bb[3];
                y1 = bb[1] + bb[3] * 2

                # create bounds for ignore regions(double the gt bbox)
                g = np.array(gt['keypoints'])
                vg = g[2::3]

                # gaidong
                dt_bb = gt['bbox']
                dt_x0 = dt_bb[0] - dt_bb[2]
                dt_x1 = dt_bb[0] + dt_bb[2] * 2
                dt_y0 = dt_bb[1] - dt_bb[3]
                dt_y1 = dt_bb[1] + dt_bb[3] * 2
                dt_w = dt_x1 - dt_x0
                dt_h = dt_y1 - dt_y0
                ref = min(dt_w, dt_h)
                dt_joints = np.array(gt['keypoints']).reshape(17, -1)
                joints_dict = map_joint_dict(dt_joints)
                # stick
                for k, link_pair in enumerate(link_pairs):
                    if link_pair[0] in joints_dict \
                            and link_pair[1] in joints_dict:
                        if dt_joints[link_pair[0], 2] < joint_thres \
                                or dt_joints[link_pair[1], 2] < joint_thres \
                                or vg[link_pair[0]] == 0 \
                                or vg[link_pair[1]] == 0:
                            continue
                    if k in range(6, 11):
                        lw = 1
                    else:
                        lw = ref / 100.
                    line = mlines.Line2D(
                        np.array([joints_dict[link_pair[0]][0],
                                  joints_dict[link_pair[1]][0]]),
                        np.array([joints_dict[link_pair[0]][1],
                                  joints_dict[link_pair[1]][1]]),
                        ls='-', lw=lw, alpha=1, color='#32CD32', )  # gt：green
                    line.set_zorder(0)
                    ax.add_line(line)
                # black ring
                for k in range(dt_joints.shape[0]):
                    if dt_joints[k, 2] < joint_thres \
                            or vg[link_pair[0]] == 0 \
                            or vg[link_pair[1]] == 0:
                        continue
                    if dt_joints[k, 0] > w or dt_joints[k, 1] > h:
                        continue
                    if k in range(5):
                        radius = 1
                    else:
                        radius = ref / 100

                    circle = mpatches.Circle(tuple(dt_joints[k, :2]),
                                             radius=radius,
                                             ec='black',
                                             fc='#32CD32',
                                             alpha=1,
                                             linewidth=1)
                    circle.set_zorder(1)
                    ax.add_patch(circle)

                for i, dt in enumerate(dts):
                    # Calculate IoU
                    dt_bb = dt['bbox']
                    dt_x0 = dt_bb[0] - dt_bb[2];
                    dt_x1 = dt_bb[0] + dt_bb[2] * 2
                    dt_y0 = dt_bb[1] - dt_bb[3];
                    dt_y1 = dt_bb[1] + dt_bb[3] * 2

                    ol_x = min(x1, dt_x1) - max(x0, dt_x0)
                    ol_y = min(y1, dt_y1) - max(y0, dt_y0)
                    ol_area = ol_x * ol_y
                    s_x = max(x1, dt_x1) - min(x0, dt_x0)
                    s_y = max(y1, dt_y1) - min(y0, dt_y0)
                    sum_area = s_x * s_y
                    iou = ol_area / (sum_area + np.spacing(1))
                    score = dt['score']

                    if iou < 0.1 or score < threshold:
                        continue
                    else:
                        print('iou: ', iou)
                        dt_w = dt_x1 - dt_x0
                        dt_h = dt_y1 - dt_y0
                        ref = min(dt_w, dt_h)
                        num_box += 1
                        sum_score += dt['score']
                        dt_joints = np.array(dt['keypoints']).reshape(17, -1)
                        joints_dict = map_joint_dict(dt_joints)

                        # stick
                        for k, link_pair in enumerate(link_pairs):
                            if link_pair[0] in joints_dict \
                                    and link_pair[1] in joints_dict:
                                if dt_joints[link_pair[0], 2] < joint_thres \
                                        or dt_joints[link_pair[1], 2] < joint_thres \
                                        or vg[link_pair[0]] == 0 \
                                        or vg[link_pair[1]] == 0:
                                    continue
                            if k in range(6, 11):
                                lw = 1
                            else:
                                lw = ref / 100.
                            line = mlines.Line2D(
                                np.array([joints_dict[link_pair[0]][0],
                                          joints_dict[link_pair[1]][0]]),
                                np.array([joints_dict[link_pair[0]][1],
                                          joints_dict[link_pair[1]][1]]),
                                ls='-', lw=lw, alpha=1, color='#00BFFF', )  # prediction：blue
                            line.set_zorder(0)
                            ax.add_line(line)
                        # black ring
                        for k in range(dt_joints.shape[0]):
                            if dt_joints[k, 2] < joint_thres \
                                    or vg[link_pair[0]] == 0 \
                                    or vg[link_pair[1]] == 0:
                                continue
                            if dt_joints[k, 0] > w or dt_joints[k, 1] > h:
                                continue
                            if k in range(5):
                                radius = 1
                            else:
                                radius = ref / 100

                            circle = mpatches.Circle(tuple(dt_joints[k, :2]),
                                                     radius=radius,
                                                     ec='black',
                                                     fc='#00BFFF',
                                                     alpha=1,
                                                     linewidth=1)
                            circle.set_zorder(1)
                            ax.add_patch(circle)

            avg_score = (sum_score / (num_box + np.spacing(1))) * 1000

            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.axis('off')
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            plt.margins(0, 0)
            if save:
                if np.int(avg_score) > 400:
                    plt.savefig(save_path + \
                                '_id_' + str(imgId) + \
                                'score_' + str(np.int(avg_score)) + \
                                '_' + img_name + '.png',
                                format='png', bbox_inckes='tight', dpi=100)
                    # plt.savefig(save_path +'id_'+str(imgId)+ '.pdf', format='pdf',
                    #             bbox_inckes='tight', dpi=100)
            # plt.show()
            plt.close()


if __name__ == '__main__':

    args = parse_args()
    if args.style == 'xiaochu':
        # Xiaochu Style
        colorstyle = xiaochu_style
    elif args.style == 'chunhua':
        # Chunhua Style
        colorstyle = chunhua_style
    else:
        raise Exception('Invalid color style')

    save_path = args.save_path
    img_path = args.image_path
    if not os.path.exists(save_path):
        try:
            os.makedirs(save_path)
        except Exception:
            print('Fail to make {}'.format(save_path))

    with open(args.prediction) as f:
        data = json.load(f)
    gt_file = args.gt_anno
    plot(data, gt_file, img_path, save_path, colorstyle.link_pairs, colorstyle.ring_color, save=True)

# --- End of .\visualization\plot_coco.py ---

