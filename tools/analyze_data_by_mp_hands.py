import _init_paths
from config import cfg
from config import update_config

import dataset
import os
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from typing import List, Dict, Tuple, Optional

# Update config from experiments
cfg.defrost()
cfg.merge_from_file("../experiments/mydataset/hrnet/w48_512x224_adam_lr1e-3.yaml")
# cfg.merge_from_list("../experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml")
cfg.DATASET.ROOT = os.path.join(
        "..", cfg.DATA_DIR, cfg.DATASET.ROOT
    )
cfg.DATASET.SCALE_FACTOR = 0.0
cfg.DATASET.ROT_FACTOR = 0
cfg.DATASET.PROB_HALF_BODY = 0.0
cfg.DATASET.NUM_JOINTS_HALF_BODY = 0
cfg.freeze()


class HandLandmarkDetector:
    """
    A class for detecting hand landmarks using MediaPipe Hands.
    """
    def __init__(self, 
                 static_image_mode=False,
                 max_num_hands=2,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        """
        Initialize the HandLandmarkDetector with MediaPipe Hands.
        
        Args:
            static_image_mode: If True, treats input images as a batch of static images.
            max_num_hands: Maximum number of hands to detect.
            min_detection_confidence: Minimum confidence for hand detection.
            min_tracking_confidence: Minimum confidence for hand tracking.
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Define hand landmark names for easier reference
        self.landmark_names = [
            "WRIST",
            "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
            "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
            "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
            "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
            "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
        ]
    
    def calculate_hand_bbox(self, landmarks, image_width, image_height, padding=10):
        """
        Calculate the bounding box for a hand based on its landmarks.
        
        Args:
            landmarks: List of hand landmarks
            image_width: Width of the image
            image_height: Height of the image
            padding: Padding to add around the hand (in pixels)
            
        Returns:
            Bounding box as [x, y, width, height]
        """
        if not landmarks:
            return None
        
        # Extract x and y coordinates
        x_coords = [landmark.x * image_width for landmark in landmarks]
        y_coords = [landmark.y * image_height for landmark in landmarks]
        
        # Calculate bounding box
        min_x = max(0, min(x_coords) - padding)
        min_y = max(0, min(y_coords) - padding)
        max_x = min(image_width, max(x_coords) + padding)
        max_y = min(image_height, max(y_coords) + padding)
        
        # Return as [x, y, width, height]
        return [min_x, min_y, max_x - min_x, max_y - min_y]
    
    def determine_hand_orientation(self, landmarks):
        """
        Determine if the hand is left or right based on the arrangement of landmarks 0, 5, 9, 13, 17.
        
        Args:
            landmarks: List of hand landmarks
            
        Returns:
            String indicating "Left Hand" or "Right Hand"
        """
        # Extract the key landmarks (wrist and MCP joints)
        key_indices = [0, 5, 9, 13, 17]
        key_landmarks = [landmarks[i] for i in key_indices]
        
        # Calculate the cross product to determine orientation
        # We need at least 3 points to calculate orientation
        x0, y0 = key_landmarks[0].x, key_landmarks[0].y  # Wrist
        x1, y1 = key_landmarks[1].x, key_landmarks[1].y  # Index MCP
        x2, y2 = key_landmarks[2].x, key_landmarks[2].y  # Middle MCP
        
        # Calculate cross product (x1-x0)*(y2-y0) - (y1-y0)*(x2-x0)
        cross_product = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
        
        # If cross product is positive, the points are arranged counterclockwise
        # If negative, they're arranged clockwise
        if cross_product < 0:
            return "Right"  # Counterclockwise -> Right hand
        else:
            return "Left"   # Clockwise -> Left hand
    
    def calculate_hand_span_ratio(self, landmarks):
        """
        Calculate the ratio of the distance between thumb tip (4) and pinky tip (20)
        to the base length (distance between middle finger MCP (9) and PIP (10)).
        
        Args:
            landmarks: List of hand landmarks
            
        Returns:
            Ratio of hand span to base length
        """
        # Extract the required landmarks
        thumb_tip = landmarks[4]
        pinky_tip = landmarks[20]
        middle_mcp = landmarks[9]
        middle_pip = landmarks[10]
        
        # Calculate distances
        def calculate_distance(p1, p2):
            return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
        
        # Base length: distance between middle finger MCP and PIP
        base_length = calculate_distance(middle_mcp, middle_pip)
        
        # Hand span: distance between thumb tip and pinky tip
        hand_span = calculate_distance(thumb_tip, pinky_tip)
        
        # Calculate ratio
        if base_length > 0:
            ratio = hand_span / base_length
        else:
            ratio = 0
            
        return ratio
    
    def detect_landmarks(self, image: np.ndarray, draw_landmark_indices=True) -> Tuple[np.ndarray, List, List, List]:
        """
        Detect hand landmarks in an image.
        
        Args:
            image: Input RGB image as numpy array.
            draw_landmark_indices: Whether to draw landmark indices on the image.
            
        Returns:
            Tuple containing:
                - Annotated image with landmarks drawn
                - List of detected hand landmarks
                - List of hand handedness (left/right)
                - List of hand bounding boxes
        """
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_height, image_width, _ = image_rgb.shape
        
        # Process the image and detect hands
        results = self.hands.process(image_rgb)
        
        # Convert back to BGR for OpenCV
        annotated_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        # Lists to store detected hand landmarks, handedness, and bboxes
        hand_landmarks_list = []
        handedness_list = []
        bbox_list = []
        
        # List to store hand orientations determined by our algorithm
        orientation_list = []
        
        # List to store hand span ratios
        span_ratio_list = []
        
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Calculate bounding box for the hand
                bbox = self.calculate_hand_bbox(hand_landmarks.landmark, image_width, image_height)
                bbox_list.append(bbox)
                
                # Determine hand orientation based on landmark arrangement
                orientation = self.determine_hand_orientation(hand_landmarks.landmark)
                orientation_list.append(orientation)
                
                # Calculate hand span ratio
                span_ratio = self.calculate_hand_span_ratio(hand_landmarks.landmark)
                span_ratio_list.append(span_ratio)
                
                # Draw the bounding box
                if bbox:
                    x, y, w, h = [int(v) for v in bbox]
                    # Get handedness for label
                    handedness = "Unknown"
                    if results.multi_handedness and idx < len(results.multi_handedness):
                        handedness = results.multi_handedness[idx].classification[0].label
                    
                    # Draw rectangle
                    cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Draw label
                    label = f"{handedness} Hand"
                    cv2.putText(annotated_image, label, (x, y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Draw the landmarks on the image
                self.mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Draw landmark indices if requested
                if draw_landmark_indices:
                    for i, landmark in enumerate(hand_landmarks.landmark):
                        # Convert normalized coordinates to pixel coordinates
                        x = int(landmark.x * image_width)
                        y = int(landmark.y * image_height)
                        
                        # Draw index number
                        cv2.putText(
                            annotated_image, 
                            str(i), 
                            (x - 10, y - 10),  # Offset to not overlap with the landmark
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5,  # Font scale
                            (255, 0, 0),  # Blue color
                            1  # Thickness
                        )
                
                # Extract landmark coordinates
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    # Convert normalized coordinates to pixel coordinates
                    x = landmark.x * image_width
                    y = landmark.y * image_height
                    z = landmark.z  # Depth value
                    landmarks.append((x, y, z))
                
                hand_landmarks_list.append(landmarks)
                
                # Get handedness (left or right hand)
                if results.multi_handedness:
                    handedness = results.multi_handedness[idx].classification[0].label
                    handedness_list.append(handedness)
                else:
                    handedness_list.append("Unknown")
        
        # Display hand orientation in the top-left corner of the image
        if orientation_list:
            # Combine all orientations if multiple hands
            orientation_text = " & ".join(orientation_list)
            cv2.putText(
                annotated_image,
                f"Orientation: {orientation_text}",
                (20, 40),  # Position in top-left
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,  # Font scale
                (0, 0, 255),  # Red color
                2  # Thickness
            )
            
            # Also display MediaPipe's handedness
            mp_handedness_text = " & ".join(handedness_list)
            cv2.putText(
                annotated_image,
                f"MediaPipe: {mp_handedness_text}",
                (20, 80),  # Position below orientation
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,  # Font scale
                (255, 0, 0),  # Blue color
                2  # Thickness
            )
            
            # Display hand span ratios
            for i, ratio in enumerate(span_ratio_list):
                cv2.putText(
                    annotated_image,
                    f"Hand {i+1} Span Ratio: {ratio:.2f}",
                    (20, 120 + i*40),  # Position below handedness
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,  # Font scale
                    (0, 128, 128),  # Teal color
                    2  # Thickness
                )
                
                # Draw lines to visualize the measurements
                if results.multi_hand_landmarks and i < len(results.multi_hand_landmarks):
                    hand_landmarks = results.multi_hand_landmarks[i]
                    
                    # Draw base length line (9-10)
                    base_start = (
                        int(hand_landmarks.landmark[9].x * image_width),
                        int(hand_landmarks.landmark[9].y * image_height)
                    )
                    base_end = (
                        int(hand_landmarks.landmark[10].x * image_width),
                        int(hand_landmarks.landmark[10].y * image_height)
                    )
                    cv2.line(annotated_image, base_start, base_end, (0, 255, 255), 2)
                    
                    # Draw hand span line (4-20)
                    span_start = (
                        int(hand_landmarks.landmark[4].x * image_width),
                        int(hand_landmarks.landmark[4].y * image_height)
                    )
                    span_end = (
                        int(hand_landmarks.landmark[20].x * image_width),
                        int(hand_landmarks.landmark[20].y * image_height)
                    )
                    cv2.line(annotated_image, span_start, span_end, (255, 0, 255), 2)
        
        return annotated_image, hand_landmarks_list, handedness_list, bbox_list, orientation_list, span_ratio_list
    
    def release(self):
        """
        Release resources.
        """
        self.hands.close()

def process_dataset_with_hand_landmarks(json_path, image_root_dir, output_dir=None):
    """
    Process all images in a dataset JSON file with hand landmark detection.
    
    Args:
        json_path: Path to the JSON annotation file
        image_root_dir: Root directory containing the images
        output_dir: Directory to save annotated images (if None, images won't be saved)
    
    Returns:
        Dictionary containing hand landmark data for each image
    """
    # Create output directory if specified
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize hand landmark detector
    detector = HandLandmarkDetector(static_image_mode=True)
    
    # Load JSON file
    print(f"Loading annotations from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract image info
    images = data.get('images', [])
    print(f"Found {len(images)} images in the dataset")
    
    # Dictionary to store results
    results_dict = {}
    
    # Process each image
    for img_info in tqdm(images, desc="Processing images"):
        img_id = img_info['id']
        img_file = img_info['file_name']
        img_path = os.path.join(image_root_dir, img_file)
        
        # Check if image exists
        if not os.path.exists(img_path):
            print(f"Warning: Image {img_path} not found. Skipping.")
            continue
        
        # Read image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Failed to read image {img_path}. Skipping.")
            continue
        
        # Detect hand landmarks with indices drawn
        annotated_image, hand_landmarks, handedness, hand_bboxes, orientations, span_ratios = detector.detect_landmarks(
            image, draw_landmark_indices=True
        )
        
        # Store results
        results_dict[img_id] = {
            'file_name': img_file,
            'hand_landmarks': hand_landmarks,
            'handedness': handedness,
            'hand_bboxes': hand_bboxes,
            'orientations': orientations,
            'span_ratios': span_ratios
        }
        
        # Save annotated image if output directory is provided
        if output_dir:
            output_path = os.path.join(output_dir, f"annotated_{os.path.basename(img_file)}")
            cv2.imwrite(output_path, annotated_image)
    
    # Release resources
    detector.release()
    
    return results_dict


def save_hand_landmarks_to_json(results_dict, output_json_path):
    """
    Save hand landmark detection results to a JSON file.
    
    Args:
        results_dict: Dictionary containing hand landmark data
        output_json_path: Path to save the JSON file
    """
    # Convert numpy arrays and other non-serializable types to lists
    serializable_dict = {}
    for img_id, data in results_dict.items():
        serializable_dict[str(img_id)] = {
            'image_name': data['file_name'],
            'hand_landmarks': [
                [[float(coord) for coord in point] for point in hand]
                for hand in data['hand_landmarks']
            ],
            'handedness': data['handedness'],
            'orientations': data.get('orientations', []),
            'span_ratios': data.get('span_ratios', []),
            'hand_bboxes': [[float(val) for val in bbox] if bbox else None for bbox in data.get('hand_bboxes', [])]
        }
    
    # Save to JSON file
    with open(output_json_path, 'w') as f:
        json.dump(serializable_dict, f, indent=2)
    
    print(f"Results saved to {output_json_path}")


def create_landmark_reference_image():
    """
    Create a reference image showing all hand landmarks with their indices.
    """
    # Create a blank image
    img_size = 800
    reference_img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
    
    # Draw hand diagram
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    # Create a mock hand landmark for visualization
    # This is a simplified representation with approximate positions
    mock_landmarks = []
    
    # Define normalized coordinates for a right hand in a flat pose
    # These are approximate positions to create a visual reference
    # WRIST - 0
    mock_landmarks.append((0.5, 0.8, 0))
    
    # THUMB - 1-4
    mock_landmarks.append((0.35, 0.75, 0))  # THUMB_CMC
    mock_landmarks.append((0.25, 0.65, 0))  # THUMB_MCP
    mock_landmarks.append((0.2, 0.55, 0))   # THUMB_IP
    mock_landmarks.append((0.15, 0.45, 0))  # THUMB_TIP
    
    # INDEX - 5-8
    mock_landmarks.append((0.45, 0.6, 0))   # INDEX_FINGER_MCP
    mock_landmarks.append((0.45, 0.45, 0))  # INDEX_FINGER_PIP
    mock_landmarks.append((0.45, 0.35, 0))  # INDEX_FINGER_DIP
    mock_landmarks.append((0.45, 0.25, 0))  # INDEX_FINGER_TIP
    
    # MIDDLE - 9-12
    mock_landmarks.append((0.5, 0.58, 0))   # MIDDLE_FINGER_MCP
    mock_landmarks.append((0.5, 0.43, 0))   # MIDDLE_FINGER_PIP
    mock_landmarks.append((0.5, 0.33, 0))   # MIDDLE_FINGER_DIP
    mock_landmarks.append((0.5, 0.23, 0))   # MIDDLE_FINGER_TIP
    
    # RING - 13-16
    mock_landmarks.append((0.55, 0.6, 0))   # RING_FINGER_MCP
    mock_landmarks.append((0.55, 0.45, 0))  # RING_FINGER_PIP
    mock_landmarks.append((0.55, 0.35, 0))  # RING_FINGER_DIP
    mock_landmarks.append((0.55, 0.25, 0))  # RING_FINGER_TIP
    
    # PINKY - 17-20
    mock_landmarks.append((0.6, 0.62, 0))   # PINKY_MCP
    mock_landmarks.append((0.6, 0.5, 0))    # PINKY_PIP
    mock_landmarks.append((0.6, 0.4, 0))    # PINKY_DIP
    mock_landmarks.append((0.6, 0.3, 0))    # PINKY_TIP
    
    # Draw connections manually
    connections = [
        # Thumb
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Index
        (0, 5), (5, 6), (6, 7), (7, 8),
        # Middle
        (0, 9), (9, 10), (10, 11), (11, 12),
        # Ring
        (0, 13), (13, 14), (14, 15), (15, 16),
        # Pinky
        (0, 17), (17, 18), (18, 19), (19, 20),
        # Palm
        (5, 9), (9, 13), (13, 17)
    ]
    
    # Draw connections
    for connection in connections:
        start_idx, end_idx = connection
        start_point = (int(mock_landmarks[start_idx][0] * img_size), 
                      int(mock_landmarks[start_idx][1] * img_size))
        end_point = (int(mock_landmarks[end_idx][0] * img_size), 
                    int(mock_landmarks[end_idx][1] * img_size))
        
        cv2.line(reference_img, start_point, end_point, (0, 0, 0), 2)
    
    # Draw landmarks with indices
    for i, (x, y, _) in enumerate(mock_landmarks):
        # Convert normalized coordinates to pixel coordinates
        px = int(x * img_size)
        py = int(y * img_size)
        
        # Draw circle for landmark
        cv2.circle(reference_img, (px, py), 8, (0, 0, 255), -1)
        
        # Draw index number
        cv2.putText(
            reference_img, 
            str(i), 
            (px - 5, py - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7,
            (0, 0, 0),
            2
        )
    
    # Add title and legend
    cv2.putText(
        reference_img,
        "Hand Landmark Reference",
        (img_size // 4, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        2
    )
    
    # Add landmark names
    landmark_names = [
        "0: WRIST",
        "1-4: THUMB (CMC, MCP, IP, TIP)",
        "5-8: INDEX FINGER (MCP, PIP, DIP, TIP)",
        "9-12: MIDDLE FINGER (MCP, PIP, DIP, TIP)",
        "13-16: RING FINGER (MCP, PIP, DIP, TIP)",
        "17-20: PINKY (MCP, PIP, DIP, TIP)"
    ]
    
    for i, name in enumerate(landmark_names):
        cv2.putText(
            reference_img,
            name,
            (50, img_size - 150 + i * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1
        )
    
    # Highlight key landmarks used for orientation detection
    key_indices = [0, 5, 9, 13, 17]
    key_points = [(int(mock_landmarks[i][0] * img_size), int(mock_landmarks[i][1] * img_size)) for i in key_indices]
    
    # Draw a polygon connecting these key points
    cv2.polylines(reference_img, [np.array(key_points)], True, (0, 255, 0), 2)
    
    # Add explanation for orientation detection
    cv2.putText(
        reference_img,
        "Hand Orientation Detection:",
        (50, img_size - 250),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        1
    )
    
    cv2.putText(
        reference_img,
        "Green polygon connects landmarks 0,5,9,13,17",
        (50, img_size - 225),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        1
    )
    
    cv2.putText(
        reference_img,
        "Clockwise arrangement = Left Hand",
        (50, img_size - 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        1
    )
    
    cv2.putText(
        reference_img,
        "Counterclockwise arrangement = Right Hand",
        (50, img_size - 175),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        1
    )
    
    # Highlight the landmarks used for hand span ratio
    # Base length: 9-10 (Middle finger MCP to PIP)
    base_start = (int(mock_landmarks[9][0] * img_size), int(mock_landmarks[9][1] * img_size))
    base_end = (int(mock_landmarks[10][0] * img_size), int(mock_landmarks[10][1] * img_size))
    cv2.line(reference_img, base_start, base_end, (0, 255, 255), 3)  # Yellow line
    
    # Hand span: 4-20 (Thumb tip to Pinky tip)
    span_start = (int(mock_landmarks[4][0] * img_size), int(mock_landmarks[4][1] * img_size))
    span_end = (int(mock_landmarks[20][0] * img_size), int(mock_landmarks[20][1] * img_size))
    cv2.line(reference_img, span_start, span_end, (255, 0, 255), 3)  # Magenta line
    
    # Add explanation for hand span ratio
    cv2.putText(
        reference_img,
        "Hand Span Ratio:",
        (50, img_size - 300),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        1
    )
    
    cv2.putText(
        reference_img,
        "Base Length (Yellow): Distance between landmarks 9-10",
        (50, img_size - 275),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        1
    )
    
    cv2.putText(
        reference_img,
        "Hand Span (Magenta): Distance between landmarks 4-20",
        (50, img_size - 250),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        1
    )
    
    cv2.putText(
        reference_img,
        "Ratio = Hand Span / Base Length",
        (50, img_size - 225),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        1
    )
    
    return reference_img


def main():
    """
    Main function to run hand landmark detection on the dataset.
    """
    # Paths
    json_path = r"C:\Users\Jack\workspace\MyProject\C2F-HumanPoseEstimation-main\data\mydataset\annotations\person_keypoints_val.json"
    
    
    # Determine image root directory based on JSON path
    image_root_dir = os.path.join(os.path.dirname(os.path.dirname(json_path)), "images", 'val')
    
    # Create output directories
    output_dir = os.path.join('./output_images', "hand_landmarks_output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Output paths
    annotated_images_dir = os.path.join(output_dir, "annotated_images")
    output_json_path = os.path.join(output_dir, "hand_landmarks_results.json")
    
    # Create a reference image for landmark indices
    reference_img = create_landmark_reference_image()
    reference_path = os.path.join(output_dir, "landmark_reference.jpg")
    cv2.imwrite(reference_path, reference_img)
    print(f"Created landmark reference image at {reference_path}")
    
    # Process dataset
    print("Starting hand landmark detection on dataset...")
    results_dict = process_dataset_with_hand_landmarks(
        json_path=json_path,
        image_root_dir=image_root_dir,
        output_dir=annotated_images_dir
    )
    
    # Save results to JSON
    save_hand_landmarks_to_json(results_dict, output_json_path)
    
    print("Hand landmark detection completed.")


if __name__ == "__main__":
    main()