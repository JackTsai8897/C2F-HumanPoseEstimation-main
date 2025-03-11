import os.path as osp
import sys

import _init_paths
from config import cfg
from config import update_config

import dataset
import torchvision.transforms as transforms
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

input, target, target_weight, meta = train_dataset[0]

# rec.append({
#                 'image': image_path,
#                 'center': center,
#                 'scale': scale,
#                 'joints_3d': joints_3d,
#                 'joints_3d_vis': joints_3d_vis,
#                 'filename': '',
#                 'imgnum': 0,
#                 'x': x,
#                 'y': y,
#                 'w': w,
#                 'h': h,
#             })

from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
import cv2
import numpy as np
import matplotlib.pyplot as plt
#image_size = np.array(cfg.MODEL.IMAGE_SIZE)
rec = train_dataset.db[0]
c = rec['center']
s = rec['scale']
r = 10
data_numpy = cv2.imread(rec['image'])
image_size = [384, 96] # model input size
# aspect ratio used in coco dataset is calculated by model input size 
# to make the transformed image has same aspect ratio as model input size, so that can avoid distortion
#image_size = np.array([384, 288])

# Draw bounding box on original image
original_image = data_numpy.copy()
x, y, w, h = rec['x'], rec['y'], rec['w'], rec['h']
cv2.rectangle(original_image, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)



trans, src, dst = get_affine_transform(c, s, r, image_size, show_pts=True,
                                        horizontal=True)
input = cv2.warpAffine(
    data_numpy,
    trans,
    (int(image_size[0]), int(image_size[1])),
    flags=cv2.INTER_LINEAR)

# Transform the bounding box coordinates
bbox_points = np.array([
    [x, y],           # top-left
    [x + w, y],       # top-right
    [x + w, y + h],   # bottom-right
    [x, y + h]        # bottom-left
])

# Transform each point of the bounding box
transformed_bbox_points = []
for point in bbox_points:
    transformed_point = affine_transform(point, trans)
    transformed_bbox_points.append(transformed_point)
transformed_bbox_points = np.array(transformed_bbox_points, dtype=np.int32)

# Draw transformed bounding box
transformed_image = input.copy()
cv2.polylines(transformed_image, [transformed_bbox_points], isClosed=True, color=(0, 255, 0), thickness=2)

# Draw src points on original image
for i, point in enumerate(src):
    cv2.circle(original_image, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)  # Red circles
    cv2.putText(original_image, f"src{i}", (int(point[0])+5, int(point[1])+5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# Draw dst points on transformed image
for i, point in enumerate(dst):
    cv2.circle(transformed_image, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)  # Red circles
    cv2.putText(transformed_image, f"dst{i}", (int(point[0])+5, int(point[1])+5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


# Save the transformed image with bounding box
cv2.imwrite('./output_images/original_with_bbox.jpg', original_image)
cv2.imwrite('./output_images/transformed_with_bbox.jpg', transformed_image)
cv2.imwrite('./output_images/test.jpg', input)

# cv2.imshow('Original Image with Bounding Box', original_image)
# cv2.imshow('Transformed Image with Bounding Box', transformed_image)
# cv2.imshow('Transformed Image', input)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# Convert BGR to RGB for matplotlib
original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
transformed_image_rgb = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)
input_rgb = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)

# Plot with matplotlib
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(original_image_rgb)
plt.title('Original Image with Bounding Box and src points')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(transformed_image_rgb)
plt.title('Transformed Image with Bounding Box and dst points')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(input_rgb)
plt.title('Transformed Image')
plt.axis('off')

plt.tight_layout()
plt.savefig('./output_images/comparison_plot.png')
plt.show()
