import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import math

# Configuration
TRAIN_DIR = "../data/mydataset/images/train"
VAL_DIR = "../data/mydataset/images/val"
OUTPUT_DIR = "visualized_results/mydataset/subjects"
OUTPUT_FILE = "subject_samples.png"
MAX_IMAGES_PER_ROW = 5

def get_subject_images(directory):
    """
    Get the first image for each subject from the directory
    
    Args:
        directory: Path to image directory
        
    Returns:
        Dictionary mapping subject names to their first image path
    """
    # Dictionary to store all images for each subject
    all_subject_images = defaultdict(list)
    
    # Check if directory exists
    if not os.path.exists(directory):
        print(f"Warning: Directory {directory} does not exist")
        return {}
    
    # Read all image files
    for filename in sorted(os.listdir(directory)):
        # Skip non-image files (simple check)
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            continue
        
        # Extract subject name from filename
        if '_img' in filename:
            subject_name = filename.split('_img')[0]
            all_subject_images[subject_name].append(os.path.join(directory, filename))
    
    # Get the first image for each subject
    subject_first_images = {}
    for subject, images in all_subject_images.items():
        if images:  # Check if the subject has any images
            subject_first_images[subject] = images[0]
    
    return subject_first_images

def plot_image_grid(images_dict, title, nrows, ncols, figsize=(20, 10)):
    """
    Plot a grid of images
    
    Args:
        images_dict: Dictionary mapping subject names to image paths
        title: Title for the plot
        nrows: Number of rows in the grid
        ncols: Number of columns in the grid
        figsize: Figure size (width, height)
        
    Returns:
        Figure object
    """
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    fig.suptitle(title, fontsize=16, y=0.98)
    
    # Flatten axes array for easier indexing
    if nrows > 1 or ncols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Sort subjects for consistent display
    subjects = sorted(images_dict.keys())
    
    # Plot each image
    for i, subject in enumerate(subjects):
        if i < len(axes):
            ax = axes[i]
            
            # Load and display image
            img_path = images_dict[subject]
            img = cv2.imread(img_path)
            if img is not None:
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Display image
                ax.imshow(img)
                ax.set_title(f"{subject}", fontsize=10)
                ax.axis('off')
            else:
                ax.text(0.5, 0.5, f"Failed to load\n{os.path.basename(img_path)}", 
                       ha='center', va='center')
                ax.axis('off')
    
    # Hide unused subplots
    for i in range(len(subjects), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig

def plot_subject_samples():
    """Plot the first image for each subject from train and validation sets"""
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get subject images from train and validation directories
    train_subject_images = get_subject_images(TRAIN_DIR)
    val_subject_images = get_subject_images(VAL_DIR)
    
    # Get all unique subject names
    all_subjects = sorted(set(list(train_subject_images.keys()) + list(val_subject_images.keys())))
    
    print(f"Found {len(all_subjects)} unique subjects")
    print(f"Train set: {len(train_subject_images)} subjects")
    print(f"Validation set: {len(val_subject_images)} subjects")
    
    # Calculate layout dimensions
    train_rows = math.ceil(len(train_subject_images) / MAX_IMAGES_PER_ROW)
    val_rows = math.ceil(len(val_subject_images) / MAX_IMAGES_PER_ROW)
    
    # Create figure with both train and val sections
    plt.figure(figsize=(20, (train_rows + val_rows) * 4 + 3))
    
    # Create separate subplots for train and val
    plt.subplot(2, 1, 1)
    plt.title('Training Set Subjects', fontsize=16)
    plt.axis('off')
    
    # Plot training set grid
    train_fig = plot_image_grid(
        train_subject_images, 
        "", 
        train_rows, 
        MAX_IMAGES_PER_ROW, 
        figsize=(18, train_rows * 4)
    )
    
    # Save train figure
    train_output = os.path.join(OUTPUT_DIR, "train_subjects.png")
    train_fig.savefig(train_output, bbox_inches='tight', dpi=150)
    plt.close(train_fig)
    
    # Plot validation set grid
    val_fig = plot_image_grid(
        val_subject_images, 
        "", 
        val_rows, 
        MAX_IMAGES_PER_ROW, 
        figsize=(18, val_rows * 4)
    )
    
    # Save validation figure
    val_output = os.path.join(OUTPUT_DIR, "val_subjects.png")
    val_fig.savefig(val_output, bbox_inches='tight', dpi=150)
    plt.close(val_fig)
    
    # Create a combined figure with both sections
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, (train_rows + val_rows) * 4 + 3),
                                  gridspec_kw={'height_ratios': [train_rows, val_rows]})
    
    # Set titles
    ax1.set_title('Training Set Subjects', fontsize=16)
    ax2.set_title('Validation Set Subjects', fontsize=16)
    
    # Turn off axes
    ax1.axis('off')
    ax2.axis('off')
    
    # Load and display the saved images
    train_img = plt.imread(train_output)
    val_img = plt.imread(val_output)
    
    ax1.imshow(train_img)
    ax2.imshow(val_img)
    
    # Add overall title
    plt.suptitle('Subject Samples from Dataset', fontsize=18, y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save combined figure
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"Subject samples saved to {output_path}")
    
    # Clean up temporary files
    os.remove(train_output)
    os.remove(val_output)

if __name__ == "__main__":
    plot_subject_samples()