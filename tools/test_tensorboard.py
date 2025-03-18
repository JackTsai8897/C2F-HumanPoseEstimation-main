import torch
from torch.utils.tensorboard import SummaryWriter

print("PyTorch version:", torch.__version__)
print("TensorBoard import successful!")

# Create a SummaryWriter instance
writer = SummaryWriter()
print("SummaryWriter created successfully!")

# Close the writer
writer.close()
print("Test completed successfully!")