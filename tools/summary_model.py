import os
import torch
import _init_paths
from config import cfg
from config import update_config
import models
from utils.utils import get_model_summary

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


cfg.MODEL.IMAGE_SIZE = [512, 224]
cfg.MODEL.HEATMAP_SIZE = [128, 56]
# cfg.MODEL.IMAGE_SIZE = [384, 288]
# cfg.MODEL.HEATMAP_SIZE = [96, 72]
cfg.freeze()

# Create the model
model, model_fine = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False)

# Create a dummy input tensor with the correct shape
# The shape should match the model's expected input
# For HRNet, it's typically [batch_size, channels, height, width]
# Based on the config name, the input size is 384x288
dummy_input = torch.randn(1, 3, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1]) # FCN can use any image size as input

# Get and print the model summary
model_summary = get_model_summary(model, dummy_input, verbose=True)
print("Main Model Summary:")
print(model_summary)

# Get and print the fine model summary if needed
fine_model_summary = get_model_summary(model_fine,
                                        torch.randn(1, cfg.MODEL.NUM_JOINTS,
                                                                cfg.MODEL.HEATMAP_SIZE[0],
                                                                  cfg.MODEL.HEATMAP_SIZE[1]), verbose=True)
print("\nFine Model Summary:")
print(fine_model_summary)