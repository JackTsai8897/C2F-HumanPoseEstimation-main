import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import re

def load_json_file(file_path):
    """Load and return JSON data from file path"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

def extract_subject_name(image_name):
    """Extract subject name from image_name using _img as delimiter"""
    if "_img" in image_name:
        return image_name.split("_img")[0]
    return "unknown"

def merge_data_by_image_name(hand_landmarks_data, per_image_results):
    """Merge data from both files based on image_name"""
    merged_data = []
    
    # Create mapping from image names to per_image_results
    image_name_to_results = {}
    
    # Process per_image_results
    for data in per_image_results:
        if "image_name" in data:
            image_name = data["image_name"]
            image_name_to_results[image_name] = data
    
    print(f"Created mapping for {len(image_name_to_results)} unique image names from per_image_results")
    
    # Print some image name examples for debugging
    print("\nSample image names from per_image_results:")
    for i, name in enumerate(list(image_name_to_results.keys())[:5]):
        print(f"  {name}")
    
    # Now merge with hand landmarks data
    match_count = 0
    
    for img_id, hand_data in hand_landmarks_data.items():
        if "image_name" in hand_data:
            image_name = hand_data["image_name"]
            subject_name = extract_subject_name(image_name)
            
            # Try to find a match
            if image_name in image_name_to_results:
                match_count += 1
                result_data = image_name_to_results[image_name]
                
                # Extract relevant data
                orientations = hand_data.get("orientations", [])
                span_ratios = hand_data.get("span_ratios", [])
                mean_distance = result_data.get("statistics", {}).get("mean_distance", None)
                
                # Extract keypoint-specific data
                keypoints_data = result_data.get("keypoints", [])
                keypoint_distances = {}
                for kp in keypoints_data:
                    kp_id = kp.get("keypoint_id")
                    kp_name = kp.get("keypoint_name")
                    kp_distance = kp.get("distance")
                    if kp_id is not None and kp_distance is not None:
                        keypoint_distances[f"keypoint_{kp_id}_distance"] = kp_distance
                        keypoint_distances[f"keypoint_{kp_name}_distance"] = kp_distance
                
                # For each hand in the image, create a record
                for i in range(len(orientations)):
                    orientation = orientations[i] if i < len(orientations) else None
                    span_ratio = span_ratios[i] if i < len(span_ratios) else None
                    
                    if orientation and span_ratio is not None and mean_distance is not None:
                        record = {
                            "image_name": image_name,
                            "image_id": img_id,
                            "subject_name": subject_name,
                            "orientation": orientation,
                            "span_ratio": span_ratio,
                            "mean_distance": mean_distance
                        }
                        # Add keypoint-specific distances
                        record.update(keypoint_distances)
                        merged_data.append(record)
    
    print(f"Found {match_count} matching images between the two datasets")
    return merged_data

def analyze_and_visualize(merged_data, output_dir="analysis_output"):
    """Analyze the merged data and create visualizations"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if merged_data is empty
    if not merged_data:
        print("No data to analyze. Please check the merge process.")
        return None
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(merged_data)
    
    # Print basic statistics
    print(f"Total merged records: {len(df)}")
    print(f"Number of unique images: {df['image_name'].nunique()}")
    print(f"Number of unique subjects: {df['subject_name'].nunique()}")
    print(f"Subjects: {', '.join(df['subject_name'].unique())}")
    print(f"Orientation distribution: {df['orientation'].value_counts().to_dict()}")
    print(f"Mean distance range: {df['mean_distance'].min():.4f} to {df['mean_distance'].max():.4f}")
    print(f"Span ratio range: {df['span_ratio'].min():.4f} to {df['span_ratio'].max():.4f}")
    
    # 1. Box plot of mean_distance by orientation
    plt.figure(figsize=(10, 6))
    plt.boxplot([df[df['orientation'] == 'Right']['mean_distance'], 
                 df[df['orientation'] == 'Left']['mean_distance']],
                labels=['Right', 'Left'])
    plt.title('Distribution of Mean Distance by Hand Orientation', fontsize=14)
    plt.xlabel('Hand Orientation', fontsize=12)
    plt.ylabel('Mean Distance', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add mean values as text
    for i, orientation in enumerate(['Right', 'Left']):
        subset = df[df['orientation'] == orientation]
        if not subset.empty:
            mean_val = subset['mean_distance'].mean()
            plt.text(i+1, mean_val, f'Mean: {mean_val:.4f}', 
                     ha='center', va='bottom', fontsize=10)
    
    plt.savefig(os.path.join(output_dir, 'mean_distance_by_orientation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Scatter plot of span_ratio vs mean_distance with regression lines
    plt.figure(figsize=(12, 8))
    
    # Plot points by orientation
    colors = {'Right': 'blue', 'Left': 'red'}
    for orientation, color in colors.items():
        subset = df[df['orientation'] == orientation]
        if not subset.empty:
            plt.scatter(subset['span_ratio'], subset['mean_distance'], 
                        label=orientation, color=color, alpha=0.7)
    
    # Add overall trend line
    if len(df) > 1:
        z = np.polyfit(df['span_ratio'], df['mean_distance'], 1)
        p = np.poly1d(z)
        x_range = np.linspace(df['span_ratio'].min(), df['span_ratio'].max(), 100)
        plt.plot(x_range, p(x_range), 'k--', label=f'Overall trend (r={df["span_ratio"].corr(df["mean_distance"]):.3f})')
    
    # Add trend lines for each orientation
    for orientation, color in colors.items():
        subset = df[df['orientation'] == orientation]
        if len(subset) > 1:
            z = np.polyfit(subset['span_ratio'], subset['mean_distance'], 1)
            p = np.poly1d(z)
            x_range = np.linspace(subset['span_ratio'].min(), subset['span_ratio'].max(), 100)
            plt.plot(x_range, p(x_range), '--', color=color, 
                     label=f'{orientation} trend (r={subset["span_ratio"].corr(subset["mean_distance"]):.3f})')
    
    plt.title('Relationship Between Hand Span Ratio and Mean Distance', fontsize=14)
    plt.xlabel('Hand Span Ratio (Thumb tip to Pinky tip / Middle MCP to PIP)', fontsize=12)
    plt.ylabel('Mean Distance', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    plt.savefig(os.path.join(output_dir, 'span_ratio_vs_mean_distance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Additional visualization: Histogram of span ratios by orientation
    plt.figure(figsize=(12, 6))
    
    for orientation, color in colors.items():
        subset = df[df['orientation'] == orientation]
        if not subset.empty:
            plt.hist(subset['span_ratio'], bins=15, alpha=0.5, label=orientation, color=color)
    
    plt.title('Distribution of Hand Span Ratios by Orientation', fontsize=14)
    plt.xlabel('Hand Span Ratio', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    plt.savefig(os.path.join(output_dir, 'span_ratio_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Calculate and print correlations
    print("\nCorrelation Analysis:")
    print(f"Overall correlation between span_ratio and mean_distance: {df['span_ratio'].corr(df['mean_distance']):.4f}")
    
    for orientation in df['orientation'].unique():
        subset = df[df['orientation'] == orientation]
        if len(subset) > 1:
            corr = subset['span_ratio'].corr(subset['mean_distance'])
            print(f"Correlation for {orientation} orientation: {corr:.4f}")
    
    # 5. Analysis for all keypoints (E0, E1, E2, W0, W1, W2)
    keypoint_analysis(df, output_dir)
    
    # 6. Save the merged data to CSV for further analysis
    csv_path = os.path.join(output_dir, 'merged_hand_data.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nMerged data saved to: {csv_path}")
    
    # 7. Subject-specific analysis
    analyze_by_subject(df, output_dir)
    
    # 8. Elbow vs Wrist comparison analysis
    elbow_vs_wrist_analysis(df, output_dir)
    
    
    return df

def keypoint_analysis(df, output_dir):
    """Analyze keypoint-specific data"""
    # Create a subdirectory for keypoint analysis
    keypoint_dir = os.path.join(output_dir, "keypoints")
    os.makedirs(keypoint_dir, exist_ok=True)
    
    # Check if keypoint data is available
    keypoint_columns = [col for col in df.columns if col.startswith('keypoint_') and col.endswith('_distance') and not col.startswith('keypoint_0_') and not col.startswith('keypoint_1_')]
    if not keypoint_columns:
        print("No keypoint-specific data found for analysis")
        return
    
    print(f"\nAnalyzing {len(keypoint_columns)} keypoint columns: {keypoint_columns}")
    
    # Get all keypoint names (E0, E1, E2, W0, W1, W2)
    all_keypoints = []
    for col in keypoint_columns:
        if col.startswith('keypoint_E') or col.startswith('keypoint_W'):
            all_keypoints.append(col)
    
    all_keypoints_available = [kp for kp in all_keypoints if kp in df.columns]
    
    if not all_keypoints_available:
        print("No keypoint data found")
        return
    
    print(f"Analyzing keypoints: {all_keypoints_available}")
    
    # 1. Box plot of keypoint distances
    plt.figure(figsize=(14, 8))
    
    # Prepare data for boxplot
    boxplot_data = []
    boxplot_labels = []
    
    for kp in all_keypoints_available:
        kp_name = kp.replace('keypoint_', '').replace('_distance', '')
        if not df[kp].empty:
            boxplot_data.append(df[kp])
            boxplot_labels.append(kp_name)
    
    if boxplot_data:
        plt.boxplot(boxplot_data, labels=boxplot_labels)
        plt.title('Distribution of Distances by Keypoint', fontsize=14)
        plt.xlabel('Keypoint', fontsize=12)
        plt.ylabel('Distance', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add mean values as text
        for i, kp in enumerate(all_keypoints_available):
            mean_val = df[kp].mean()
            plt.text(i+1, mean_val, f'Mean: {mean_val:.2f}', 
                    ha='center', va='bottom', fontsize=10)
        
        plt.savefig(os.path.join(keypoint_dir, 'keypoint_distances_boxplot.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Scatter plots of span_ratio vs keypoint distances
    for kp in all_keypoints_available:
        kp_name = kp.replace('keypoint_', '').replace('_distance', '')
        
        plt.figure(figsize=(10, 6))
        
        # Plot points by orientation
        for orientation, color in {'Right': 'blue', 'Left': 'red'}.items():
            subset = df[df['orientation'] == orientation]
            if not subset.empty:
                plt.scatter(subset['span_ratio'], subset[kp], 
                           label=orientation, color=color, alpha=0.7)
        
        # Add overall trend line
        if len(df) > 1:
            z = np.polyfit(df['span_ratio'], df[kp], 1)
            p = np.poly1d(z)
            x_range = np.linspace(df['span_ratio'].min(), df['span_ratio'].max(), 100)
            plt.plot(x_range, p(x_range), 'k--', 
                    label=f'Overall trend (r={df["span_ratio"].corr(df[kp]):.3f})')
        
        plt.title(f'Span Ratio vs {kp_name} Distance', fontsize=14)
        plt.xlabel('Hand Span Ratio', fontsize=12)
        plt.ylabel(f'{kp_name} Distance', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        plt.savefig(os.path.join(keypoint_dir, f'span_ratio_vs_{kp_name}_distance.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Print correlation statistics
    print("\nKeypoint Correlation Analysis:")
    for kp in all_keypoints_available:
        kp_name = kp.replace('keypoint_', '').replace('_distance', '')
        corr = df['span_ratio'].corr(df[kp])
        print(f"Correlation between span_ratio and {kp_name} distance: {corr:.4f}")
        
        # By orientation
        for orientation in df['orientation'].unique():
            subset = df[df['orientation'] == orientation]
            if len(subset) > 1:
                corr = subset['span_ratio'].corr(subset[kp])
                print(f"  {orientation} orientation: {corr:.4f}")
    
    # 4. Compare E keypoints vs W keypoints
    e_keypoints = [kp for kp in all_keypoints_available if 'keypoint_E' in kp]
    w_keypoints = [kp for kp in all_keypoints_available if 'keypoint_W' in kp]
    
    if e_keypoints and w_keypoints:
        # Create average distance columns for E and W keypoints
        df['avg_E_distance'] = df[e_keypoints].mean(axis=1)
        df['avg_W_distance'] = df[w_keypoints].mean(axis=1)
        
        # Box plot comparing E vs W keypoints
        plt.figure(figsize=(10, 6))
        plt.boxplot([df['avg_E_distance'], df['avg_W_distance']], 
                   labels=['E Keypoints (Elbow)', 'W Keypoints (Wrist)'])
        plt.title('Comparison of E vs W Keypoint Distances', fontsize=14)
        plt.xlabel('Keypoint Group', fontsize=12)
        plt.ylabel('Average Distance', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add mean values as text
        for i, col in enumerate(['avg_E_distance', 'avg_W_distance']):
            mean_val = df[col].mean()
            plt.text(i+1, mean_val, f'Mean: {mean_val:.2f}', 
                    ha='center', va='bottom', fontsize=10)
        
        plt.savefig(os.path.join(keypoint_dir, 'e_vs_w_keypoints_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Scatter plot of span ratio vs average E and W distances
        plt.figure(figsize=(12, 8))
        plt.scatter(df['span_ratio'], df['avg_E_distance'], label='E Keypoints', color='blue', alpha=0.7)
        plt.scatter(df['span_ratio'], df['avg_W_distance'], label='W Keypoints', color='red', alpha=0.7)
        
        # Add trend lines
        for col, color, label in [('avg_E_distance', 'blue', 'E Keypoints'), 
                                 ('avg_W_distance', 'red', 'W Keypoints')]:
            z = np.polyfit(df['span_ratio'], df[col], 1)
            p = np.poly1d(z)
            x_range = np.linspace(df['span_ratio'].min(), df['span_ratio'].max(), 100)
            plt.plot(x_range, p(x_range), '--', color=color, 
                    label=f'{label} trend (r={df["span_ratio"].corr(df[col]):.3f})')
        
        plt.title('Span Ratio vs Average Keypoint Distances', fontsize=14)
        plt.xlabel('Hand Span Ratio', fontsize=12)
        plt.ylabel('Average Distance', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        plt.savefig(os.path.join(keypoint_dir, 'span_ratio_vs_avg_distances.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print correlation statistics for average distances
        print("\nAverage Keypoint Group Correlation Analysis:")
        print(f"Correlation between span_ratio and avg E distance: {df['span_ratio'].corr(df['avg_E_distance']):.4f}")
        print(f"Correlation between span_ratio and avg W distance: {df['span_ratio'].corr(df['avg_W_distance']):.4f}")

def elbow_vs_wrist_analysis(df, output_dir):
    """Compare Elbow vs Wrist keypoints against span ratio"""
    # Create a subdirectory for this analysis
    ew_dir = os.path.join(output_dir, "elbow_vs_wrist")
    os.makedirs(ew_dir, exist_ok=True)
    
    # Check if we have the keypoint columns
    e_keypoints = [col for col in df.columns if col.startswith('keypoint_E') and col.endswith('_distance')]
    w_keypoints = [col for col in df.columns if col.startswith('keypoint_W') and col.endswith('_distance')]
    
    if not e_keypoints or not w_keypoints:
        print("No elbow or wrist keypoint columns found for comparison")
        return
    
    print("\nAnalyzing Elbow vs Wrist keypoint performance...")
    
    # Calculate average distances for elbow and wrist keypoints
    df['avg_elbow_distance'] = df[e_keypoints].mean(axis=1)
    df['avg_wrist_distance'] = df[w_keypoints].mean(axis=1)
    
    # 1. Scatter plot of span_ratio vs avg_elbow_distance and avg_wrist_distance
    plt.figure(figsize=(12, 8))
    
    # Plot points for elbow keypoints
    plt.scatter(df['span_ratio'], df['avg_elbow_distance'], 
                label='Elbow Keypoints (E0, E1, E2)', color='darkgreen', alpha=0.7, marker='o')
    
    # Plot points for wrist keypoints
    plt.scatter(df['span_ratio'], df['avg_wrist_distance'], 
                label='Wrist Keypoints (W0, W1, W2)', color='darkblue', alpha=0.7, marker='x')
    
    # Add trend lines
    # For elbow keypoints
    z_elbow = np.polyfit(df['span_ratio'], df['avg_elbow_distance'], 1)
    p_elbow = np.poly1d(z_elbow)
    x_range = np.linspace(df['span_ratio'].min(), df['span_ratio'].max(), 100)
    elbow_corr = df['span_ratio'].corr(df['avg_elbow_distance'])
    plt.plot(x_range, p_elbow(x_range), '-', color='green', 
             label=f'Elbow trend (r={elbow_corr:.3f})')
    
    # For wrist keypoints
    z_wrist = np.polyfit(df['span_ratio'], df['avg_wrist_distance'], 1)
    p_wrist = np.poly1d(z_wrist)
    wrist_corr = df['span_ratio'].corr(df['avg_wrist_distance'])
    plt.plot(x_range, p_wrist(x_range), '-', color='blue', 
             label=f'Wrist trend (r={wrist_corr:.3f})')
    
    plt.title('Span Ratio vs Average Keypoint Distance by Type', fontsize=14)
    plt.xlabel('Hand Span Ratio', fontsize=12)
    plt.ylabel('Average Distance (pixels)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    plt.savefig(os.path.join(ew_dir, 'span_ratio_vs_elbow_wrist_distance.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Box plot comparison of elbow vs wrist distances
    plt.figure(figsize=(10, 6))
    
    plt.boxplot([df['avg_elbow_distance'], df['avg_wrist_distance']], 
               labels=['Elbow Keypoints (E0, E1, E2)', 'Wrist Keypoints (W0, W1, W2)'])
    
    plt.title('Distribution of Average Distances by Keypoint Type', fontsize=14)
    plt.xlabel('Keypoint Type', fontsize=12)
    plt.ylabel('Average Distance (pixels)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add mean values as text
    for i, col in enumerate(['avg_elbow_distance', 'avg_wrist_distance']):
        mean_val = df[col].mean()
        plt.text(i+1, mean_val, f'Mean: {mean_val:.2f}', 
                ha='center', va='bottom', fontsize=10)
    
    plt.savefig(os.path.join(ew_dir, 'elbow_vs_wrist_boxplot.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Analyze by orientation
    for orientation in df['orientation'].unique():
        subset = df[df['orientation'] == orientation]
        if len(subset) < 3:
            continue
            
        # Scatter plot for this orientation
        plt.figure(figsize=(12, 8))
        
        # Plot points
        plt.scatter(subset['span_ratio'], subset['avg_elbow_distance'], 
                    label='Elbow Keypoints (E0, E1, E2)', color='darkgreen', alpha=0.7, marker='o')
        plt.scatter(subset['span_ratio'], subset['avg_wrist_distance'], 
                    label='Wrist Keypoints (W0, W1, W2)', color='darkblue', alpha=0.7, marker='x')
        
        # Add trend lines
        # For elbow keypoints
        z_elbow = np.polyfit(subset['span_ratio'], subset['avg_elbow_distance'], 1)
        p_elbow = np.poly1d(z_elbow)
        x_range = np.linspace(subset['span_ratio'].min(), subset['span_ratio'].max(), 100)
        elbow_corr = subset['span_ratio'].corr(subset['avg_elbow_distance'])
        plt.plot(x_range, p_elbow(x_range), '-', color='green', 
                 label=f'Elbow trend (r={elbow_corr:.3f})')
        
        # For wrist keypoints
        z_wrist = np.polyfit(subset['span_ratio'], subset['avg_wrist_distance'], 1)
        p_wrist = np.poly1d(z_wrist)
        wrist_corr = subset['span_ratio'].corr(subset['avg_wrist_distance'])
        plt.plot(x_range, p_wrist(x_range), '-', color='blue', 
                 label=f'Wrist trend (r={wrist_corr:.3f})')
        
        plt.title(f'{orientation} Hand: Span Ratio vs Average Keypoint Distance by Type', fontsize=14)
        plt.xlabel('Hand Span Ratio', fontsize=12)
        plt.ylabel('Average Distance (pixels)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        plt.savefig(os.path.join(ew_dir, f'span_ratio_vs_elbow_wrist_distance_{orientation}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Print statistics
    print("\nElbow vs Wrist Keypoint Statistics:")
    print(f"Average Elbow Keypoint Distance: {df['avg_elbow_distance'].mean():.4f} (±{df['avg_elbow_distance'].std():.4f})")
    print(f"Average Wrist Keypoint Distance: {df['avg_wrist_distance'].mean():.4f} (±{df['avg_wrist_distance'].std():.4f})")
    print(f"Correlation between span_ratio and avg_elbow_distance: {df['span_ratio'].corr(df['avg_elbow_distance']):.4f}")
    print(f"Correlation between span_ratio and avg_wrist_distance: {df['span_ratio'].corr(df['avg_wrist_distance']):.4f}")
    
    # 5. Calculate difference between elbow and wrist distances
    df['elbow_wrist_diff'] = df['avg_elbow_distance'] - df['avg_wrist_distance']
    
    # Scatter plot of span ratio vs difference
    plt.figure(figsize=(12, 8))
    
    # Plot points by orientation
    colors = {'Right': 'purple', 'Left': 'orange'}
    for orientation, color in colors.items():
        subset = df[df['orientation'] == orientation]
        if not subset.empty:
            plt.scatter(subset['span_ratio'], subset['elbow_wrist_diff'], 
                        label=orientation, color=color, alpha=0.7)
    
    # Add overall trend line
    z = np.polyfit(df['span_ratio'], df['elbow_wrist_diff'], 1)
    p = np.poly1d(z)
    x_range = np.linspace(df['span_ratio'].min(), df['span_ratio'].max(), 100)
    diff_corr = df['span_ratio'].corr(df['elbow_wrist_diff'])
    plt.plot(x_range, p(x_range), 'k--', label=f'Overall trend (r={diff_corr:.3f})')
    
    plt.title('Span Ratio vs Difference Between Elbow and Wrist Distances', fontsize=14)
    plt.xlabel('Hand Span Ratio', fontsize=12)
    plt.ylabel('Elbow Distance - Wrist Distance (pixels)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # Add horizontal line at y=0
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    plt.savefig(os.path.join(ew_dir, 'span_ratio_vs_elbow_wrist_difference.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Create individual plots for each keypoint
    # Create a subdirectory for individual keypoint analysis
    individual_dir = os.path.join(ew_dir, "individual_keypoints")
    os.makedirs(individual_dir, exist_ok=True)
    
    # Plot each elbow keypoint separately
    for kp in e_keypoints:
        kp_name = kp.replace('keypoint_', '').replace('_distance', '')
        
        plt.figure(figsize=(10, 6))
        
        # Plot points by orientation
        for orientation, color in {'Right': 'darkgreen', 'Left': 'lightgreen'}.items():
            subset = df[df['orientation'] == orientation]
            if not subset.empty:
                plt.scatter(subset['span_ratio'], subset[kp], 
                           label=f'{orientation} {kp_name}', color=color, alpha=0.7)
        
        # Add overall trend line
        z = np.polyfit(df['span_ratio'], df[kp], 1)
        p = np.poly1d(z)
        x_range = np.linspace(df['span_ratio'].min(), df['span_ratio'].max(), 100)
        corr = df['span_ratio'].corr(df[kp])
        plt.plot(x_range, p(x_range), 'g--', 
                label=f'Trend (r={corr:.3f})')
        
        plt.title(f'Span Ratio vs {kp_name} Distance', fontsize=14)
        plt.xlabel('Hand Span Ratio', fontsize=12)
        plt.ylabel(f'{kp_name} Distance (pixels)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        plt.savefig(os.path.join(individual_dir, f'span_ratio_vs_{kp_name}_distance.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot each wrist keypoint separately
    for kp in w_keypoints:
        kp_name = kp.replace('keypoint_', '').replace('_distance', '')
        
        plt.figure(figsize=(10, 6))
        
        # Plot points by orientation
        for orientation, color in {'Right': 'darkblue', 'Left': 'lightblue'}.items():
            subset = df[df['orientation'] == orientation]
            if not subset.empty:
                plt.scatter(subset['span_ratio'], subset[kp], 
                           label=f'{orientation} {kp_name}', color=color, alpha=0.7)
        
        # Add overall trend line
        z = np.polyfit(df['span_ratio'], df[kp], 1)
        p = np.poly1d(z)
        x_range = np.linspace(df['span_ratio'].min(), df['span_ratio'].max(), 100)
        corr = df['span_ratio'].corr(df[kp])
        plt.plot(x_range, p(x_range), 'b--', 
                label=f'Trend (r={corr:.3f})')
        
        plt.title(f'Span Ratio vs {kp_name} Distance', fontsize=14)
        plt.xlabel('Hand Span Ratio', fontsize=12)
        plt.ylabel(f'{kp_name} Distance (pixels)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        plt.savefig(os.path.join(individual_dir, f'span_ratio_vs_{kp_name}_distance.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # 7. Combined plot with all individual keypoints
    plt.figure(figsize=(14, 10))
    
    # Plot each elbow keypoint
    colors_e = ['darkgreen', 'green', 'lightgreen']
    for i, kp in enumerate(e_keypoints):
        kp_name = kp.replace('keypoint_', '').replace('_distance', '')
        plt.scatter(df['span_ratio'], df[kp], 
                   label=kp_name, color=colors_e[i % len(colors_e)], alpha=0.6, marker='o')
    
    # Plot each wrist keypoint
    colors_w = ['darkblue', 'blue', 'lightblue']
    for i, kp in enumerate(w_keypoints):
        kp_name = kp.replace('keypoint_', '').replace('_distance', '')
        plt.scatter(df['span_ratio'], df[kp], 
                   label=kp_name, color=colors_w[i % len(colors_w)], alpha=0.6, marker='x')
    
    # Add trend lines for average elbow and wrist
    z_elbow = np.polyfit(df['span_ratio'], df['avg_elbow_distance'], 1)
    p_elbow = np.poly1d(z_elbow)
    x_range = np.linspace(df['span_ratio'].min(), df['span_ratio'].max(), 100)
    plt.plot(x_range, p_elbow(x_range), '-', color='darkgreen', linewidth=2,
             label=f'Avg Elbow trend (r={elbow_corr:.3f})')
    
    z_wrist = np.polyfit(df['span_ratio'], df['avg_wrist_distance'], 1)
    p_wrist = np.poly1d(z_wrist)
    plt.plot(x_range, p_wrist(x_range), '-', color='darkblue', linewidth=2,
             label=f'Avg Wrist trend (r={wrist_corr:.3f})')
    
    plt.title('Span Ratio vs Individual Keypoint Distances', fontsize=14)
    plt.xlabel('Hand Span Ratio', fontsize=12)
    plt.ylabel('Distance (pixels)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(ew_dir, 'span_ratio_vs_all_keypoints.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Elbow vs Wrist analysis complete. Results saved to {ew_dir}")
    
def analyze_by_subject(df, output_dir):
    """Analyze data by individual subjects"""
    # Create a subdirectory for subject-specific analysis
    subject_dir = os.path.join(output_dir, "subjects")
    os.makedirs(subject_dir, exist_ok=True)
    
    # Get unique subjects
    subjects = df['subject_name'].unique()
    print(f"\nAnalyzing data for {len(subjects)} individual subjects...")
    
    # Create a summary dataframe for subject statistics
    subject_stats = []
    
    # Print hand orientation counts by subject
    print("\nHand orientation counts by subject:")
    orientation_counts = df.groupby(['subject_name', 'orientation']).size().unstack(fill_value=0)
    print(orientation_counts)

    # Calculate totals
    total_left = orientation_counts['Left'].sum() if 'Left' in orientation_counts.columns else 0
    total_right = orientation_counts['Right'].sum() if 'Right' in orientation_counts.columns else 0

    # Create a bar chart of hand orientation counts by subject
    plt.figure(figsize=(12, 8))

    # Set up the bar positions
    x = np.arange(len(orientation_counts.index))
    width = 0.35

    # Create bars for Left and Right hands
    left_counts = orientation_counts['Left'] if 'Left' in orientation_counts.columns else np.zeros(len(orientation_counts.index))
    right_counts = orientation_counts['Right'] if 'Right' in orientation_counts.columns else np.zeros(len(orientation_counts.index))

    plt.bar(x - width/2, left_counts, width, label=f'Left Hand (Total: {total_left})', color='orange', alpha=0.7)
    plt.bar(x + width/2, right_counts, width, label=f'Right Hand (Total: {total_right})', color='purple', alpha=0.7)

    # Add labels and title
    plt.xlabel('Subject', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.title('Hand Orientation Counts by Subject', fontsize=14)
    plt.xticks(x, orientation_counts.index, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, linestyle='--', axis='y', alpha=0.7)

    # Add count labels on top of bars
    for i, (left, right) in enumerate(zip(left_counts, right_counts)):
        if left > 0:
            plt.text(i - width/2, left + 0.5, str(int(left)), ha='center', va='bottom')
        if right > 0:
            plt.text(i + width/2, right + 0.5, str(int(right)), ha='center', va='bottom')

    # Add total counts as text annotation in the upper part of the plot
    plt.annotate(f'Total Left: {total_left}, Total Right: {total_right}', 
                xy=(0.5, 0.95), xycoords='axes fraction',
                ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))

    plt.tight_layout()
    plt.savefig(os.path.join(subject_dir, 'hand_orientation_counts_by_subject.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

    # Save the orientation counts to CSV
    orientation_counts.to_csv(os.path.join(subject_dir, 'hand_orientation_counts.csv'))
    
    # Check for keypoint columns
    keypoint_columns = [col for col in df.columns if col.startswith('keypoint_') and col.endswith('_distance') and not col.startswith('keypoint_0_') and not col.startswith('keypoint_1_')]
    all_keypoints = []
    for col in keypoint_columns:
        if col.startswith('keypoint_E') or col.startswith('keypoint_W'):
            all_keypoints.append(col)
    
    all_keypoints_available = [kp for kp in all_keypoints if kp in df.columns]
    
    # Analyze each subject
    for subject in subjects:
        # Filter data for this subject
        subject_data = df[df['subject_name'] == subject]
        
        # Skip if too few data points
        if len(subject_data) < 3:
            print(f"Skipping subject {subject}: insufficient data ({len(subject_data)} records)")
            continue
            
        print(f"Analyzing subject: {subject} ({len(subject_data)} records)")
        
        # Calculate statistics
        mean_distance = subject_data['mean_distance'].mean()
        median_distance = subject_data['mean_distance'].median()
        mean_span_ratio = subject_data['span_ratio'].mean()
        
        # Calculate keypoint statistics
        keypoint_stats = {}
        for kp in all_keypoints_available:
            kp_name = kp.replace('keypoint_', '').replace('_distance', '')
            keypoint_stats[f'{kp_name}_mean'] = subject_data[kp].mean()
            keypoint_stats[f'{kp_name}_median'] = subject_data[kp].median()
            keypoint_stats[f'{kp_name}_correlation'] = subject_data['span_ratio'].corr(subject_data[kp])
        
        # Count orientations
        right_count = len(subject_data[subject_data['orientation'] == 'Right'])
        left_count = len(subject_data[subject_data['orientation'] == 'Left'])
        
        # Calculate correlation
        correlation = subject_data['span_ratio'].corr(subject_data['mean_distance'])
        
        # Add to summary stats
        subject_stat = {
            'subject_name': subject,
            'record_count': len(subject_data),
            'mean_distance': mean_distance,
            'median_distance': median_distance,
            'mean_span_ratio': mean_span_ratio,
            'right_hand_count': right_count,
            'left_hand_count': left_count,
            'span_distance_correlation': correlation
        }
        subject_stat.update(keypoint_stats)
        subject_stats.append(subject_stat)
        
        # 1. Box plot of mean_distance by orientation for this subject
        plt.figure(figsize=(8, 5))
        
        # Create boxplot data
        boxplot_data = []
        boxplot_labels = []
        
        for orientation in ['Right', 'Left']:
            orient_data = subject_data[subject_data['orientation'] == orientation]
            if len(orient_data) > 0:
                boxplot_data.append(orient_data['mean_distance'])
                boxplot_labels.append(orientation)
        
        if boxplot_data:
            plt.boxplot(boxplot_data, labels=boxplot_labels)
            plt.title(f'Subject: {subject} - Mean Distance by Hand Orientation', fontsize=14)
            plt.xlabel('Hand Orientation', fontsize=12)
            plt.ylabel('Mean Distance', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Add mean values as text
            for i, orientation in enumerate(['Right', 'Left']):
                subset = subject_data[subject_data['orientation'] == orientation]
                if not subset.empty:
                    mean_val = subset['mean_distance'].mean()
                    plt.text(i+1, mean_val, f'Mean: {mean_val:.4f}', 
                            ha='center', va='bottom', fontsize=10)
            
            plt.savefig(os.path.join(subject_dir, f'{subject}_mean_distance_by_orientation.png'), 
                        dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Scatter plot of span_ratio vs mean_distance for this subject
        plt.figure(figsize=(10, 6))

        # Plot points by orientation
        colors = {'Right': 'blue', 'Left': 'red'}
        for orientation, color in colors.items():
            subset = subject_data[subject_data['orientation'] == orientation]
            if not subset.empty:
                plt.scatter(subset['span_ratio'], subset['mean_distance'], 
                            label=orientation, color=color, alpha=0.7)
                
                # Add trend line for each orientation if enough data
                if len(subset) > 2:
                    z = np.polyfit(subset['span_ratio'], subset['mean_distance'], 1)
                    p = np.poly1d(z)
                    x_range = np.linspace(subset['span_ratio'].min(), subset['span_ratio'].max(), 100)
                    orient_corr = subset['span_ratio'].corr(subset['mean_distance'])
                    plt.plot(x_range, p(x_range), '--', color=color, 
                            label=f'{orientation} trend (r={orient_corr:.3f})')

        # Add overall trend line if enough data
        if len(subject_data) > 2:
            z = np.polyfit(subject_data['span_ratio'], subject_data['mean_distance'], 1)
            p = np.poly1d(z)
            x_range = np.linspace(subject_data['span_ratio'].min(), subject_data['span_ratio'].max(), 100)
            plt.plot(x_range, p(x_range), 'k--', 
                    label=f'Overall trend (r={correlation:.3f})')

        plt.title(f'Subject: {subject} - Span Ratio vs Mean Distance', fontsize=14)
        plt.xlabel('Hand Span Ratio', fontsize=12)
        plt.ylabel('Mean Distance', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        plt.savefig(os.path.join(subject_dir, f'{subject}_span_ratio_vs_mean_distance.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Keypoint analysis for this subject
        if all_keypoints_available:
            # Create a subdirectory for this subject's keypoint analysis
            subject_keypoint_dir = os.path.join(subject_dir, f"{subject}_keypoints")
            os.makedirs(subject_keypoint_dir, exist_ok=True)
            
            # Box plot of keypoint distances
            plt.figure(figsize=(14, 8))
            
            # Prepare data for boxplot
            boxplot_data = []
            boxplot_labels = []
            
            for kp in all_keypoints_available:
                kp_name = kp.replace('keypoint_', '').replace('_distance', '')
                if not subject_data[kp].empty:
                    boxplot_data.append(subject_data[kp])
                    boxplot_labels.append(kp_name)
            
            if boxplot_data:
                plt.boxplot(boxplot_data, labels=boxplot_labels)
                plt.title(f'Subject: {subject} - Distribution of Distances by Keypoint', fontsize=14)
                plt.xlabel('Keypoint', fontsize=12)
                plt.ylabel('Distance', fontsize=12)
                plt.grid(True, linestyle='--', alpha=0.7)
                
                # Add mean values as text
                for i, kp in enumerate(all_keypoints_available):
                    mean_val = subject_data[kp].mean()
                    plt.text(i+1, mean_val, f'Mean: {mean_val:.2f}', 
                            ha='center', va='bottom', fontsize=10)
                
                plt.savefig(os.path.join(subject_keypoint_dir, f'{subject}_keypoint_distances_boxplot.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
            
            # Scatter plots for each keypoint
            for kp in all_keypoints_available:
                kp_name = kp.replace('keypoint_', '').replace('_distance', '')
                
                plt.figure(figsize=(10, 6))
                
                # Plot points by orientation
                for orientation, color in colors.items():
                    subset = subject_data[subject_data['orientation'] == orientation]
                    if not subset.empty:
                        plt.scatter(subset['span_ratio'], subset[kp], 
                                   label=orientation, color=color, alpha=0.7)
                        
                        # Add trend line for each orientation if enough data
                        if len(subset) > 2:
                            z = np.polyfit(subset['span_ratio'], subset[kp], 1)
                            p = np.poly1d(z)
                            x_range = np.linspace(subset['span_ratio'].min(), subset['span_ratio'].max(), 100)
                            orient_corr = subset['span_ratio'].corr(subset[kp])
                            plt.plot(x_range, p(x_range), '--', color=color, 
                                    label=f'{orientation} trend (r={orient_corr:.3f})')
                
                # Add overall trend line if enough data
                if len(subject_data) > 2:
                    corr = subject_data['span_ratio'].corr(subject_data[kp])
                    z = np.polyfit(subject_data['span_ratio'], subject_data[kp], 1)
                    p = np.poly1d(z)
                    x_range = np.linspace(subject_data['span_ratio'].min(), subject_data['span_ratio'].max(), 100)
                    plt.plot(x_range, p(x_range), 'k--', 
                            label=f'Overall trend (r={corr:.3f})')
                
                plt.title(f'Subject: {subject} - Span Ratio vs {kp_name} Distance', fontsize=14)
                plt.xlabel('Hand Span Ratio', fontsize=12)
                plt.ylabel(f'{kp_name} Distance', fontsize=12)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend(fontsize=10)
                plt.savefig(os.path.join(subject_keypoint_dir, f'{subject}_span_ratio_vs_{kp_name}_distance.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
            
            # Compare E vs W keypoints for this subject
            e_keypoints = [kp for kp in all_keypoints_available if 'keypoint_E' in kp]
            w_keypoints = [kp for kp in all_keypoints_available if 'keypoint_W' in kp]
            
            if e_keypoints and w_keypoints:
                # Create average distance columns for E and W keypoints
                subject_data['avg_E_distance'] = subject_data[e_keypoints].mean(axis=1)
                subject_data['avg_W_distance'] = subject_data[w_keypoints].mean(axis=1)
                
                # Box plot comparing E vs W keypoints
                plt.figure(figsize=(10, 6))
                plt.boxplot([subject_data['avg_E_distance'], subject_data['avg_W_distance']], 
                           labels=['E Keypoints (Elbow)', 'W Keypoints (Wrist)'])
                plt.title(f'Subject: {subject} - Comparison of E vs W Keypoint Distances', fontsize=14)
                plt.xlabel('Keypoint Group', fontsize=12)
                plt.ylabel('Average Distance', fontsize=12)
                plt.grid(True, linestyle='--', alpha=0.7)
                
                # Add mean values as text
                for i, col in enumerate(['avg_E_distance', 'avg_W_distance']):
                    mean_val = subject_data[col].mean()
                    plt.text(i+1, mean_val, f'Mean: {mean_val:.2f}', 
                            ha='center', va='bottom', fontsize=10)
                
                plt.savefig(os.path.join(subject_keypoint_dir, f'{subject}_e_vs_w_keypoints_comparison.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
    
    # Create summary table of subject statistics
    if subject_stats:
        subject_df = pd.DataFrame(subject_stats)
        
        # Save to CSV
        subject_df.to_csv(os.path.join(subject_dir, 'subject_summary.csv'), index=False)
        
        # Create a summary plot comparing subjects
        plt.figure(figsize=(12, 8))
        
        # Sort by mean distance
        subject_df_sorted = subject_df.sort_values('mean_distance')
        
        # Bar chart of mean distances by subject
        plt.bar(subject_df_sorted['subject_name'], subject_df_sorted['mean_distance'],
                color='skyblue', alpha=0.7)
        
        plt.title('Mean Distance by Subject', fontsize=14)
        plt.xlabel('Subject', fontsize=12)
        plt.ylabel('Mean Distance', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, linestyle='--', axis='y', alpha=0.7)
        
        # Add record count as text
        for i, (_, row) in enumerate(subject_df_sorted.iterrows()):
            plt.text(i, row['mean_distance'] + 0.5, 
                     f"n={row['record_count']}", 
                     ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(subject_dir, 'subject_comparison.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create comparison plots for keypoint statistics
        if all_keypoints_available:
            # Group keypoints by E and W
            e_keypoints_names = [kp.replace('keypoint_', '').replace('_distance', '') for kp in all_keypoints_available if 'keypoint_E' in kp]
            w_keypoints_names = [kp.replace('keypoint_', '').replace('_distance', '') for kp in all_keypoints_available if 'keypoint_W' in kp]
            
            # Create bar charts for each keypoint's mean distance by subject
            for kp_name in e_keypoints_names + w_keypoints_names:
                kp_mean_col = f'{kp_name}_mean'
                
                if kp_mean_col in subject_df.columns:
                    plt.figure(figsize=(12, 8))
                    
                    # Sort by the keypoint's mean distance
                    subject_df_sorted_kp = subject_df.sort_values(kp_mean_col)
                    
                    # Bar chart
                    color = 'lightgreen' if kp_name.startswith('E') else 'lightblue'
                    plt.bar(subject_df_sorted_kp['subject_name'], subject_df_sorted_kp[kp_mean_col],
                            color=color, alpha=0.7)
                    
                    plt.title(f'{kp_name} Mean Distance by Subject', fontsize=14)
                    plt.xlabel('Subject', fontsize=12)
                    plt.ylabel(f'{kp_name} Mean Distance', fontsize=12)
                    plt.xticks(rotation=45, ha='right')
                    plt.grid(True, linestyle='--', axis='y', alpha=0.7)
                    
                    # Add record count as text
                    for i, (_, row) in enumerate(subject_df_sorted_kp.iterrows()):
                        plt.text(i, row[kp_mean_col] + 0.5, 
                                f"n={row['record_count']}", 
                                ha='center', va='bottom', fontsize=9)
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(subject_dir, f'{kp_name}_subject_comparison.png'), 
                                dpi=300, bbox_inches='tight')
                    plt.close()
            
            # Create comparison of E vs W keypoint averages across subjects
            if e_keypoints_names and w_keypoints_names:
                # Calculate average E and W metrics for each subject
                for subject_idx, subject_row in subject_df.iterrows():
                    e_means = [subject_row[f'{kp}_mean'] for kp in e_keypoints_names if f'{kp}_mean' in subject_df.columns]
                    w_means = [subject_row[f'{kp}_mean'] for kp in w_keypoints_names if f'{kp}_mean' in subject_df.columns]
                    
                    if e_means:
                        subject_df.at[subject_idx, 'avg_E_mean'] = sum(e_means) / len(e_means)
                    if w_means:
                        subject_df.at[subject_idx, 'avg_W_mean'] = sum(w_means) / len(w_means)
                
                # Create bar chart comparing E vs W keypoints across subjects
                if 'avg_E_mean' in subject_df.columns and 'avg_W_mean' in subject_df.columns:
                    plt.figure(figsize=(14, 8))
                    
                    # Sort by average E keypoint distance
                    subject_df_sorted = subject_df.sort_values('avg_E_mean')
                    
                    # Set up bar positions
                    x = np.arange(len(subject_df_sorted))
                    width = 0.35
                    
                    # Create bars
                    plt.bar(x - width/2, subject_df_sorted['avg_E_mean'], width, label='E Keypoints', color='lightgreen', alpha=0.7)
                    plt.bar(x + width/2, subject_df_sorted['avg_W_mean'], width, label='W Keypoints', color='lightblue', alpha=0.7)
                    
                    plt.title('Average E vs W Keypoint Distances by Subject', fontsize=14)
                    plt.xlabel('Subject', fontsize=12)
                    plt.ylabel('Average Distance', fontsize=12)
                    plt.xticks(x, subject_df_sorted['subject_name'], rotation=45, ha='right')
                    plt.legend(fontsize=10)
                    plt.grid(True, linestyle='--', axis='y', alpha=0.7)
                    
                    # Add record count as text
                    for i, (_, row) in enumerate(subject_df_sorted.iterrows()):
                        plt.text(i, max(row['avg_E_mean'], row['avg_W_mean']) + 0.5, 
                                f"n={row['record_count']}", 
                                ha='center', va='bottom', fontsize=9)
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(subject_dir, 'e_vs_w_keypoints_by_subject.png'), 
                                dpi=300, bbox_inches='tight')
                    plt.close()
        
        print(f"Subject-specific analysis complete. Results saved to {subject_dir}")
    else:
        print("No subject-specific analysis could be performed due to insufficient data.")

def main():
    # Define file paths
    hand_landmarks_file = r"C:\Users\Jack\workspace\MyProject\C2F-HumanPoseEstimation-main\tools\output_images\hand_landmarks_output\hand_landmarks_results.json"
    per_image_results_file = r"C:\Users\Jack\workspace\MyProject\C2F-HumanPoseEstimation-main\output\mydataset\pose_hrnet\new_w48_512x224_adam_lr1e-3\per_image_results.json"
    output_dir = "./output_images/hand_landmarks_output"
    
    # Load data from both files
    print(f"Loading hand landmarks data from: {hand_landmarks_file}")
    hand_landmarks_data = load_json_file(hand_landmarks_file)
    
    print(f"Loading per-image results from: {per_image_results_file}")
    per_image_results = load_json_file(per_image_results_file)
    
    if not hand_landmarks_data or not per_image_results:
        print("Failed to load one or both data files. Exiting.")
        return
    
    print(f"Hand landmarks data contains {len(hand_landmarks_data)} entries")
    print(f"Per-image results contains {len(per_image_results)} entries")
    
    # Print sample entries to verify structure
    print("\nSample hand_landmarks_data entry:")
    sample_key = next(iter(hand_landmarks_data))
    print(f"Key: {sample_key}")
    sample_entry = hand_landmarks_data[sample_key]
    print(f"Keys in entry: {list(sample_entry.keys())}")
    
    print("\nSample per_image_results entry:")
    if per_image_results:
        sample_entry = per_image_results[0]
        print(f"Keys in entry: {list(sample_entry.keys())}")
        if "statistics" in sample_entry:
            print(f"Statistics keys: {list(sample_entry['statistics'].keys())}")
    
    # Debug: Print a few image_name values from both files to check for matches
    print("\nSample image names from hand_landmarks_data:")
    count = 0
    for key, data in hand_landmarks_data.items():
        if "image_name" in data and count < 5:
            print(f"  {data['image_name']}")
            count += 1
    
    print("\nSample image names from per_image_results:")
    count = 0
    for data in per_image_results:
        if "image_name" in data and count < 5:
            print(f"  {data['image_name']}")
            count += 1
    
    # Try to find a pattern in the image names
    print("\nAnalyzing image name patterns:")
    hand_landmarks_pattern = None
    per_image_pattern = None
    
    # Check for patterns in hand_landmarks_data
    if hand_landmarks_data:
        sample_names = []
        for key, data in hand_landmarks_data.items():
            if "image_name" in data and len(sample_names) < 3:
                sample_names.append(data["image_name"])
        
        if sample_names:
            print(f"Hand landmarks image name examples: {', '.join(sample_names)}")
            # Look for common patterns
            if all("img" in name for name in sample_names):
                hand_landmarks_pattern = "Contains 'img'"
            if all(".png" in name for name in sample_names):
                hand_landmarks_pattern = "Contains '.png'"
            print(f"Detected pattern: {hand_landmarks_pattern}")
    
    # Check for patterns in per_image_results
    if per_image_results:
        sample_names = []
        for data in per_image_results[:3]:
            if "image_name" in data:
                sample_names.append(data["image_name"])
        
        if sample_names:
            print(f"Per-image results image name examples: {', '.join(sample_names)}")
            # Look for common patterns
            if all("img" in name for name in sample_names):
                per_image_pattern = "Contains 'img'"
            if all(".png" in name for name in sample_names):
                per_image_pattern = "Contains '.png'"
            print(f"Detected pattern: {per_image_pattern}")
    
    # Merge the data
    print("\nMerging data based on image_name...")
    merged_data = merge_data_by_image_name(hand_landmarks_data, per_image_results)
    print(f"Created {len(merged_data)} merged records")
    
    if not merged_data:
        print("No data was merged. Please check if the image_name fields match between the two files.")
        
        # Suggest a manual mapping approach
        print("\nSuggestion: You may need to manually map between the two datasets.")
        print("Here's how the image names differ between the two datasets:")
        
        # Show a few examples from each dataset
        print("\nFirst few hand_landmarks_data image names:")
        for i, (key, data) in enumerate(hand_landmarks_data.items()):
            if i >= 5: break
            if "image_name" in data:
                print(f"  {data['image_name']}")
        
        print("\nFirst few per_image_results image names:")
        for i, data in enumerate(per_image_results):
            if i >= 5: break
            if "image_name" in data:
                print(f"  {data['image_name']}")
        
        return
    
    # Analyze and visualize the merged data
    print("Analyzing data and creating visualizations...")
    df = analyze_and_visualize(merged_data, output_dir)
    
    if df is not None:
        print(f"Analysis complete! Results saved to {output_dir}")
    else:
        print("Analysis could not be completed due to insufficient data.")

if __name__ == "__main__":
    main()