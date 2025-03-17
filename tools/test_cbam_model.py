import torch

import _init_paths
from models.pose_hrnet import Bottleneck_cbam
from utils.utils import get_model_summary

num_joints = 6
model_fine = Bottleneck_cbam(num_joints,num_joints)

# dummy_input = torch.randn(1, 3, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1]) # FCN can use any image size as input

# Get and print the model summary
fine_model_summary = get_model_summary(model_fine,
                                        torch.randn(1, num_joints, 128, 56), verbose=True)
print("\nFine Model Summary:")
print(fine_model_summary)