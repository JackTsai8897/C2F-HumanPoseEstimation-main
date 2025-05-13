import cv2
import mediapipe as mp
import numpy as np
from typing import List, Dict, Tuple, Optional


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
    
    def detect_landmarks(self, image: np.ndarray, draw_landmark_indices=True) -> Tuple[np.ndarray, List, List, List, List, List]:
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
                - List of hand orientations
                - List of hand span ratios
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