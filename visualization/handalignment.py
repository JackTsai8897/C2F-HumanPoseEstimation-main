import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Tuple, Dict, Optional
from pathlib import Path

class HandAlignmentProcessor:
    """
    A class for aligning hand images based on detected landmarks.
    This class uses landmarks from HandLandmarkDetector to align hand images
    to a reference template.
    """
    def __init__(self, output_size: Tuple[int, int] = (256, 256), 
                 reference_landmarks: Optional[List] = None,
                 reference_image: Optional[np.ndarray] = None,
                 alignment_method: str = 'affine',
                 key_landmarks: List[int] = None):
        """
        Initialize the HandAlignmentProcessor.
        
        Args:
            output_size: Size of the output aligned images (width, height)
            reference_landmarks: Optional predefined reference landmarks to align to
            reference_image: Optional reference image to use as template
            alignment_method: Method for alignment ('affine', 'similarity', 'perspective')
            key_landmarks: List of landmark indices to use for alignment (default uses all)
        """
        self.output_size = output_size
        self.reference_landmarks = reference_landmarks
        self.reference_image = reference_image
        self.alignment_method = alignment_method
        
        # If no key landmarks specified, use a default set that works well for alignment
        if key_landmarks is None:
            # Using wrist, index MCP, pinky MCP, middle finger tip as default key points
            # These points define a good reference frame for the hand
            self.key_landmarks = [0, 5, 17, 12]
        else:
            self.key_landmarks = key_landmarks
            
        # Create output directory
        os.makedirs('visualization/alignment_results', exist_ok=True)
        
        # Standard landmark names from MediaPipe
        self.landmark_names = [
            "WRIST",
            "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
            "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
            "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
            "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
            "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
        ]
    
    def set_reference_from_first_image(self, image: np.ndarray, landmarks: List[Tuple[float, float, float]]) -> None:
        """
        Set the reference template from the first image and its landmarks.
        
        Args:
            image: First image to use as reference
            landmarks: Landmarks detected in the first image
        """
        # Store the reference image at its original size
        self.reference_image = image.copy()
        
        # Store the original landmarks directly
        self.reference_landmarks = landmarks.copy()
        
        # Create and save a visualization of the reference template
        self._visualize_reference_template(self.reference_landmarks)
        
        # Save the reference image
        cv2.imwrite('visualization/alignment_results/reference_image.png', self.reference_image)
        
        # Update output size to match the reference image
        self.output_size = (image.shape[1], image.shape[0])
        
        print("Reference template set from the first image with original landmarks")
    
    def create_reference_template(self, landmarks_list: List[List[Tuple[float, float, float]]],
                                 handedness_list: List[str] = None) -> np.ndarray:
        """
        Create a reference template from a list of hand landmarks.
        
        Args:
            landmarks_list: List of hand landmarks from multiple images
            handedness_list: List of handedness (Left/Right) for each set of landmarks
            
        Returns:
            Reference landmarks for alignment
        """
        if not landmarks_list:
            print("No landmarks provided to create reference template")
            return None
        
        # Filter landmarks by handedness if provided
        filtered_landmarks = []
        if handedness_list:
            for landmarks, handedness in zip(landmarks_list, handedness_list):
                # We might want to create separate templates for left and right hands
                # For now, we'll use all hands to create a single template
                filtered_landmarks.append(landmarks)
        else:
            filtered_landmarks = landmarks_list
        
        if not filtered_landmarks:
            print("No landmarks left after filtering by handedness")
            return None
        
        # Calculate average position for each landmark
        num_landmarks = len(filtered_landmarks[0])
        reference = []
        
        for i in range(num_landmarks):
            x_sum = 0
            y_sum = 0
            z_sum = 0
            count = 0
            
            for landmarks in filtered_landmarks:
                if i < len(landmarks):
                    x, y, z = landmarks[i]
                    x_sum += x
                    y_sum += y
                    z_sum += z
                    count += 1
            
            if count > 0:
                reference.append((x_sum/count, y_sum/count, z_sum/count))
            else:
                reference.append((0, 0, 0))
        
        # Normalize the reference template to fit within output_size
        reference = self._normalize_landmarks(reference)
        
        # Create and save a visualization of the reference template
        self._visualize_reference_template(reference)
        
        return reference
    
    def _normalize_landmarks(self, landmarks: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        """
        Normalize landmarks to fit within the output size.
        
        Args:
            landmarks: List of (x, y, z) landmark coordinates
            
        Returns:
            Normalized landmarks
        """
        # Extract x and y coordinates
        x_coords = [lm[0] for lm in landmarks]
        y_coords = [lm[1] for lm in landmarks]
        
        # Find min and max values
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        # Calculate scaling factors
        width = max_x - min_x
        height = max_y - min_y
        
        # Add padding (10% on each side)
        padding_x = width * 0.1
        padding_y = height * 0.1
        
        scale_x = (self.output_size[0] * 0.8) / width if width > 0 else 1
        scale_y = (self.output_size[1] * 0.8) / height if height > 0 else 1
        
        # Use the smaller scale to maintain aspect ratio
        scale = min(scale_x, scale_y)
        
        # Calculate offsets to center the hand
        offset_x = (self.output_size[0] - width * scale) / 2
        offset_y = (self.output_size[1] - height * scale) / 2
        
        # Normalize landmarks
        normalized = []
        for x, y, z in landmarks:
            new_x = (x - min_x) * scale + offset_x
            new_y = (y - min_y) * scale + offset_y
            normalized.append((new_x, new_y, z))
        
        return normalized
    
    def _visualize_reference_template(self, reference_landmarks: List[Tuple[float, float, float]]) -> None:
        """
        Create a visualization of the reference template.
        
        Args:
            reference_landmarks: Reference landmarks to visualize
        """
        # Create a blank image or use the reference image if available
        if self.reference_image is not None:
            template_image = self.reference_image.copy()
        else:
            template_image = np.zeros((self.output_size[1], self.output_size[0], 3), dtype=np.uint8)
        
        # Draw landmarks
        for i, (x, y, _) in enumerate(reference_landmarks):
            cv2.circle(template_image, (int(x), int(y)), 3, (0, 255, 0), -1)
            
            # Add landmark index and name
            if i < len(self.landmark_names):
                name = self.landmark_names[i]
            else:
                name = f"Point {i}"
                
            cv2.putText(template_image, f"{i}: {name}", (int(x) + 5, int(y) + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Draw connections between landmarks (similar to MediaPipe hand connections)
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (0, 5), (5, 9), (9, 13), (13, 17)  # Palm
        ]
        
        for connection in connections:
            start_idx, end_idx = connection
            if start_idx < len(reference_landmarks) and end_idx < len(reference_landmarks):
                start_point = (int(reference_landmarks[start_idx][0]), int(reference_landmarks[start_idx][1]))
                end_point = (int(reference_landmarks[end_idx][0]), int(reference_landmarks[end_idx][1]))
                cv2.line(template_image, start_point, end_point, (0, 255, 255), 1)
        
        # Highlight key landmarks used for alignment
        for idx in self.key_landmarks:
            if idx < len(reference_landmarks):
                x, y, _ = reference_landmarks[idx]
                cv2.circle(template_image, (int(x), int(y)), 5, (0, 0, 255), -1)
        
        # Save the template image
        cv2.imwrite('visualization/alignment_results/reference_template.png', template_image)
        
    def align_image(self, image: np.ndarray, landmarks: List[Tuple[float, float, float]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align a hand image based on its landmarks to the reference template.
        
        Args:
            image: Input image containing the hand
            landmarks: Hand landmarks detected in the image
            
        Returns:
            Tuple of (aligned_image, transformation_matrix)
        """
        if self.reference_landmarks is None:
            print("Reference landmarks not set. Creating from the current landmarks.")
            self.set_reference_from_first_image(image, landmarks)
            return image, np.eye(3)  # Return identity transformation for the first image
        
        # Extract key landmarks for alignment
        src_points = []
        dst_points = []
        
        for idx in self.key_landmarks:
            if idx < len(landmarks) and idx < len(self.reference_landmarks):
                src_x, src_y, _ = landmarks[idx]
                dst_x, dst_y, _ = self.reference_landmarks[idx]
                
                src_points.append([src_x, src_y])
                dst_points.append([dst_x, dst_y])
        
        # Convert to numpy arrays
        src_points = np.array(src_points, dtype=np.float32)
        dst_points = np.array(dst_points, dtype=np.float32)
        
        # Check if we have enough points
        if len(src_points) < 3:
            print("Not enough valid landmarks for alignment")
            return image, None
        
        # Calculate transformation matrix based on selected method
        if self.alignment_method == 'affine':
            # Affine transformation (preserves parallel lines)
            transformation_matrix = cv2.getAffineTransform(src_points[:3], dst_points[:3])
            # Create a 3x3 matrix from the 2x3 affine matrix for landmark transformation
            full_transform = np.eye(3)
            full_transform[:2, :] = transformation_matrix
            
            aligned_image = cv2.warpAffine(
                image, transformation_matrix, self.output_size,
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
            )
            return aligned_image, full_transform
            
        elif self.alignment_method == 'similarity':
            # Similarity transformation (preserves angles)
            transformation_matrix = cv2.estimateAffinePartial2D(src_points, dst_points)[0]
            # Create a 3x3 matrix from the 2x3 similarity matrix
            full_transform = np.eye(3)
            full_transform[:2, :] = transformation_matrix
            
            aligned_image = cv2.warpAffine(
                image, transformation_matrix, self.output_size,
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
            )
            return aligned_image, full_transform
            
        elif self.alignment_method == 'perspective':
            # Perspective transformation (more flexible but may distort)
            if len(src_points) >= 4:
                transformation_matrix = cv2.getPerspectiveTransform(
                    src_points[:4].astype(np.float32), 
                    dst_points[:4].astype(np.float32)
                )
                aligned_image = cv2.warpPerspective(
                    image, transformation_matrix, self.output_size,
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
                )
                return aligned_image, transformation_matrix
            else:
                print("Need at least 4 points for perspective transformation")
                return image, None
        else:
            print(f"Unknown alignment method: {self.alignment_method}")
            return image, None
    
    def transform_landmarks(self, landmarks: List[Tuple[float, float, float]], 
                           transformation_matrix: np.ndarray) -> List[Tuple[float, float, float]]:
        """
        Transform landmarks using the same transformation matrix applied to the image.
        
        Args:
            landmarks: Original landmarks
            transformation_matrix: Transformation matrix used for image alignment
            
        Returns:
            Transformed landmarks
        """
        transformed_landmarks = []
        
        for x, y, z in landmarks:
            if self.alignment_method in ['affine', 'similarity']:
                # For affine transformation
                point = np.array([x, y, 1.0])
                new_x, new_y = np.dot(transformation_matrix, point)[:2]
                transformed_landmarks.append((new_x, new_y, z))
            elif self.alignment_method == 'perspective':
                # For perspective transformation
                point = np.array([x, y, 1.0])
                point_transformed = np.dot(transformation_matrix, point)
                new_x = point_transformed[0] / point_transformed[2]
                new_y = point_transformed[1] / point_transformed[2]
                transformed_landmarks.append((new_x, new_y, z))
        
        return transformed_landmarks
    
    def visualize_alignment(self, original_image: np.ndarray, aligned_image: np.ndarray,
                           original_landmarks: List[Tuple[float, float, float]],
                           transformed_landmarks: List[Tuple[float, float, float]],
                           output_path: str) -> None:
        """
        Visualize the alignment process by showing the original and aligned images side by side.
        
        Args:
            original_image: Original input image
            aligned_image: Aligned output image
            original_landmarks: Original landmarks
            transformed_landmarks: Transformed landmarks
            output_path: Path to save the visualization
        """
        # Create figure with 3 subplots: original, aligned, and reference (if available)
        if self.reference_image is not None:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig_cols = 3
        else:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            fig_cols = 2
        
        # Show original image with landmarks
        axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original Image")
        
        # Draw original landmarks
        for i, (x, y, _) in enumerate(original_landmarks):
            axes[0].scatter(x, y, c='r', s=20)
            
            # Highlight key landmarks
            if i in self.key_landmarks:
                axes[0].scatter(x, y, c='g', s=50, alpha=0.5)
                axes[0].text(x+5, y+5, str(i), fontsize=8, color='white',
                            bbox=dict(facecolor='black', alpha=0.5))
        
        # Show aligned image with transformed landmarks
        axes[1].imshow(cv2.cvtColor(aligned_image, cv2.COLOR_BGR2RGB))
        axes[1].set_title("Aligned Image")
        
        # Draw transformed landmarks
        for i, (x, y, _) in enumerate(transformed_landmarks):
            axes[1].scatter(x, y, c='r', s=20)
            
            # Highlight key landmarks
            if i in self.key_landmarks:
                axes[1].scatter(x, y, c='g', s=50, alpha=0.5)
                axes[1].text(x+5, y+5, str(i), fontsize=8, color='white',
                            bbox=dict(facecolor='black', alpha=0.5))
        
        # Show reference image if available
        if fig_cols == 3 and self.reference_image is not None:
            axes[2].imshow(cv2.cvtColor(self.reference_image, cv2.COLOR_BGR2RGB))
            axes[2].set_title("Reference Template")
            
            # Draw reference landmarks
            for i, (x, y, _) in enumerate(self.reference_landmarks):
                axes[2].scatter(x, y, c='r', s=20)
                
                # Highlight key landmarks
                if i in self.key_landmarks:
                    axes[2].scatter(x, y, c='g', s=50, alpha=0.5)
                    axes[2].text(x+5, y+5, str(i), fontsize=8, color='white',
                                bbox=dict(facecolor='black', alpha=0.5))
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    def process_video_frame(self, frame: np.ndarray, landmarks: List[Tuple[float, float, float]]) -> np.ndarray:
        """
        Process a single video frame for real-time alignment.
        
        Args:
            frame: Input video frame
            landmarks: Hand landmarks detected in the frame
            
        Returns:
            Aligned frame
        """
        if not landmarks:
            return frame
        
        # Align the frame
        aligned_frame, transformation_matrix = self.align_image(frame, landmarks)
        
        if transformation_matrix is None:
            return frame
        
        return aligned_frame
    
    def batch_process_images(self, images: List[np.ndarray], 
                            landmarks_list: List[List[Tuple[float, float, float]]],
                            output_dir: str = 'visualization/alignment_results/aligned_images',
                            visualize: bool = True) -> List[np.ndarray]:
        """
        Process a batch of images for alignment.
        
        Args:
            images: List of input images
            landmarks_list: List of landmarks for each image
            output_dir: Directory to save aligned images
            visualize: Whether to create visualization of the alignment
            
        Returns:
            List of aligned images
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Use the first image as reference if not already set
        if self.reference_landmarks is None and images and landmarks_list:
            print("Setting reference from the first image")
            self.set_reference_from_first_image(images[0], landmarks_list[0])
        
        aligned_images = []
        
        for i, (image, landmarks) in enumerate(zip(images, landmarks_list)):
            # Skip the first image if it's the reference
            if i == 0 and self.reference_image is not None:
                # The first image is already the reference, so just resize it to match output_size
                aligned_image = cv2.resize(image, self.output_size)
                aligned_images.append(aligned_image)
                
                # Save the aligned (resized) first image
                output_path = os.path.join(output_dir, f"aligned_{i}.png")
                cv2.imwrite(output_path, aligned_image)
                
                # Create visualization if requested
                if visualize:
                    # For the first image, the transformation is just resizing
                    transformed_landmarks = self._normalize_landmarks(landmarks)
                    
                    # Create and save visualization
                    vis_output_path = os.path.join(output_dir, f"alignment_vis_{i}.png")
                    self.visualize_alignment(
                        image, aligned_image, landmarks, transformed_landmarks, vis_output_path
                    )
                
                continue
            
            # Align the image
            aligned_image, transformation_matrix = self.align_image(image, landmarks)
            
            if transformation_matrix is None:
                print(f"Could not align image {i}")
                aligned_images.append(image)  # Use original if alignment fails
                continue
            
            aligned_images.append(aligned_image)
            
            # Save the aligned image
            output_path = os.path.join(output_dir, f"aligned_{i}.png")
            cv2.imwrite(output_path, aligned_image)
            
            # Create visualization if requested
            if visualize:
                # Transform landmarks to match the aligned image
                transformed_landmarks = self.transform_landmarks(landmarks, transformation_matrix)
                
                # Create and save visualization
                vis_output_path = os.path.join(output_dir, f"alignment_vis_{i}.png")
                self.visualize_alignment(
                    image, aligned_image, landmarks, transformed_landmarks, vis_output_path
                )
        
        return aligned_images


def demo_hand_alignment():
    """
    Demonstrate the hand alignment functionality using a webcam.
    """
    from hand_landmark_detector import HandLandmarkDetector
    
    # Initialize the hand landmark detector
    detector = HandLandmarkDetector(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    
    # Initialize the hand alignment processor
    aligner = HandAlignmentProcessor(
        output_size=(512, 512),
        alignment_method='similarity',
        # Use wrist, index MCP, pinky MCP, and middle finger tip for alignment
        key_landmarks=[0, 5, 17, 12]
    )
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    # Create output directory
    os.makedirs('visualization/alignment_results/demo', exist_ok=True)
    
    frame_count = 0
    reference_set = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect hand landmarks
        annotated_image, hand_landmarks_list, handedness_list, _, _, _ = detector.detect_landmarks(frame)
        
        # If hands detected
        if hand_landmarks_list:
            # Use the first hand detected
            landmarks = hand_landmarks_list[0]
            
            # Set reference landmarks from the first good frame
            if not reference_set:
                aligner.set_reference_from_first_image(frame, landmarks)
                reference_set = True
                print("Reference template created from first frame")
            
            # Align the frame
            aligned_frame, transformation_matrix = aligner.align_image(frame, landmarks)
            
            if transformation_matrix is not None:
                # Transform landmarks to match the aligned image
                transformed_landmarks = aligner.transform_landmarks(landmarks, transformation_matrix)
                
                # Display the aligned frame
                cv2.imshow('Aligned Hand', aligned_frame)
                
                # Save every 30th frame for visualization
                if frame_count % 30 == 0:
                    # Save the original and aligned frames
                    cv2.imwrite(f'visualization/alignment_results/demo/original_{frame_count}.png', frame)
                    cv2.imwrite(f'visualization/alignment_results/demo/aligned_{frame_count}.png', aligned_frame)
                    
                    # Create and save visualization
                    vis_output_path = f'visualization/alignment_results/demo/alignment_vis_{frame_count}.png'
                    aligner.visualize_alignment(
                        frame, aligned_frame, landmarks, transformed_landmarks, vis_output_path
                    )
                    print(f"Saved visualization for frame {frame_count}")
        
        # Display the annotated image
        cv2.imshow('Hand Tracking', annotated_image)
        
        # Increment frame counter
        frame_count += 1
        
        # Break loop on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    detector.release()
    
    print("Demo completed")


def batch_process_example():
    """
    Example of batch processing a set of images with hand landmarks.
    """
    from hand_landmark_detector import HandLandmarkDetector
    import glob
    
    # Initialize the hand landmark detector
    detector = HandLandmarkDetector(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.7
    )
    
    # Initialize the hand alignment processor
    aligner = HandAlignmentProcessor(
        output_size=(1920, 1080),
        alignment_method='similarity'
    )
    
    # Get list of image files
    image_files = glob.glob(r'C:\Users\Jack\workspace\MyProject\C2F-HumanPoseEstimation-main\data\mydataset\visualize\train\left\*.png')
    
    if not image_files:
        print("No images found. Please specify the correct path.")
        return
    
    # Process each image
    images = []
    landmarks_list = []
    
    for image_file in image_files:
        # Read image
        image = cv2.imread(image_file)
        
        if image is None:
            print(f"Could not read image: {image_file}")
            continue
        
        # Detect hand landmarks
        _, hand_landmarks_list, _, _, _, _ = detector.detect_landmarks(image)
        
        if not hand_landmarks_list:
            print(f"No hands detected in: {image_file}")
            continue
        
        # Use the first hand detected
        landmarks = hand_landmarks_list[0]
        
        # Add to lists
        images.append(image)
        landmarks_list.append(landmarks)
    
    # Batch process the images
    aligned_images = aligner.batch_process_images(
        images, landmarks_list, 
        output_dir='visualization/alignment_results/batch_aligned',
        visualize=True
    )
    
    print(f"Processed {len(aligned_images)} images")
    
    # Release resources
    detector.release()


if __name__ == "__main__":
    # Run the demo
    # demo_hand_alignment()
    
    # Uncomment to run the batch processing example
    batch_process_example()