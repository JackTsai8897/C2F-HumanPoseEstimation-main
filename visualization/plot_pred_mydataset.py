import json
import os
import cv2
import numpy as np

# Configuration
JSON_FILE = "../output/mydataset/pose_hrnet/new_w48_512x224_adam_lr1e-3/per_image_results.json"  # Path to your JSON file
GT_JSON_FILE = "../data/mydataset/annotations/person_keypoints_val.json"  # Path to your ground truth JSON file
IMAGE_DIR = "../data/mydataset/images/val"               # Directory containing your images
OUTPUT_DIR = "visualized_results/mydataset/val_pred"      # Directory to save visualizations
SHOW_LABELS = True                 # Whether to show keypoint labels
SHOW_DISTANCE = False             # Whether to show distance between points
SHOW_BBOX = True                  # Whether to show bounding boxes

def load_json_data(json_file):
    """Load keypoints data from JSON file"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def get_bbox_by_image_name(gt_data, image_name):
    """
    Extract bounding boxes for a specific image from ground truth data
    
    Args:
        gt_data: Ground truth data from COCO format JSON
        image_name: Image filename to find bounding boxes for
        
    Returns:
        List of bounding boxes in format [x, y, width, height]
    """
    # Find image id by image name
    image_id = None
    for img in gt_data['images']:
        if img['file_name'] == image_name:
            image_id = img['id']
            break
    
    if image_id is None:
        return []
    
    # Find annotations for this image
    bboxes = []
    for ann in gt_data['annotations']:
        if ann['image_id'] == image_id:
            bboxes.append(ann['bbox'])
    
    return bboxes

def visualize_keypoints():
    """Visualize keypoints on images"""
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load JSON data
    data = load_json_data(JSON_FILE)
    
    # Load ground truth data for bounding boxes
    gt_data = load_json_data(GT_JSON_FILE)
    
    # Define colors for visualization
    colors = {
        'pred': (0, 0, 255),  # Red for predictions (BGR)
        'gt': (0, 255, 0),    # Green for ground truth (BGR)
        'bbox': (255, 165, 0) # Blue for bounding boxes (BGR)
    }
    
    # Process each image
    for item in data:
        image_name = item['image_name']
        image_path = os.path.join(IMAGE_DIR, image_name)
        
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found. Skipping...")
            continue
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Failed to load image {image_path}. Skipping...")
            continue
        
        # Create a copy for visualization
        vis_image = image.copy()
        
        # Get image dimensions for text placement
        h, w = image.shape[:2]
        
        # Add image name and statistics
        stats = item['statistics']
        info_text = [
            f"Image: {image_name}",
            f"Mean distance: {stats['mean_distance']:.2f}px",
            f"Median distance: {stats['median_distance']:.2f}px",
            f"Max distance: {stats['max_distance']:.2f}px",
            f"Min distance: {stats['min_distance']:.2f}px",
            f"Valid keypoints: {stats['valid_keypoints']}/{stats['total_keypoints']}"
        ]
        
        # Add text to image
        for i, text in enumerate(info_text):
            cv2.putText(vis_image, text, (10, 30 + i * 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw bounding boxes if requested
        if SHOW_BBOX:
            # Get bounding boxes for this image
            bboxes = get_bbox_by_image_name(gt_data, image_name)
            
            # Draw each bounding box
            for i, bbox in enumerate(bboxes):
                # COCO format bbox is [x, y, width, height]
                x, y, w, h = [int(v) for v in bbox]
                
                # Draw rectangle
                cv2.rectangle(vis_image, (x, y), (x + w, y + h), colors['bbox'], 2)
                
                # Add bbox label
                cv2.putText(vis_image, f"bbox {i+1}", (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['bbox'], 2)
        
        # Draw keypoints
        for kp in item['keypoints']:
            # Extract keypoint information
            kp_id = kp['keypoint_id']
            kp_name = kp['keypoint_name']
            pred = kp['predicted']
            gt = kp['ground_truth']
            distance = kp['distance']
            
            # Convert coordinates to integers
            pred_x, pred_y = int(pred[0]), int(pred[1])
            gt_x, gt_y = int(gt[0]), int(gt[1])
            
            # Draw predicted keypoint (red)
            cv2.circle(vis_image, (pred_x, pred_y), 5, colors['pred'], -1)
            
            # Draw ground truth keypoint (green)
            cv2.circle(vis_image, (gt_x, gt_y), 5, colors['gt'], -1)
            
            # Draw line between predicted and ground truth
            cv2.line(vis_image, (pred_x, pred_y), (gt_x, gt_y), (255, 255, 0), 1)
            
            # Add labels if requested
            if SHOW_LABELS:
                # Label for predicted point
                cv2.putText(vis_image, f"{kp_name}", (pred_x + 10, pred_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['pred'], 2)
                
                # Label for ground truth point
                cv2.putText(vis_image, f"{kp_name}", (gt_x + 10, gt_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['gt'], 2)
            
            # Add distance if requested
            if SHOW_DISTANCE:
                mid_x = (pred_x + gt_x) // 2
                mid_y = (pred_y + gt_y) // 2
                cv2.putText(vis_image, f"{distance:.1f}px", (mid_x, mid_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Add legend
        legend_y = h - 100  # Moved up to make room for bbox legend
        
        # Predicted keypoint legend
        cv2.circle(vis_image, (20, legend_y), 5, colors['pred'], -1)
        cv2.putText(vis_image, "Predicted", (30, legend_y + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Ground truth keypoint legend
        cv2.circle(vis_image, (20, legend_y + 30), 5, colors['gt'], -1)
        cv2.putText(vis_image, "Ground Truth", (30, legend_y + 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Bounding box legend
        if SHOW_BBOX:
            cv2.rectangle(vis_image, (15, legend_y + 55), (25, legend_y + 65), colors['bbox'], 2)
            cv2.putText(vis_image, "Bounding Box", (30, legend_y + 65), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Save the visualization
        output_path = os.path.join(OUTPUT_DIR, f"vis_{image_name}")
        cv2.imwrite(output_path, vis_image)
        print(f"Saved visualization to {output_path}")
    
    print(f"Visualization complete. Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    visualize_keypoints()