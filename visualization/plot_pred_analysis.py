import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import matplotlib.patches as patches

# Configuration
JSON_FILE = "../output/mydataset/pose_hrnet/new_w48_512x224_adam_lr1e-3/per_image_results.json"
OUTPUT_DIR = "visualized_results/mydataset/analysis"
OUTPUT_CSV = "keypoint_error_analysis.csv"
OUTPUT_PLOT = "keypoint_error_analysis.png"
OUTPUT_HEATMAP = "keypoint_error_heatmap.png"
OUTPUT_BOXPLOT = "keypoint_error_boxplot.png"

def load_json_data(json_file):
    """Load keypoints data from JSON file"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def analyze_keypoint_errors():
    """Analyze keypoint errors by subject and generate statistics"""
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load JSON data
    data = load_json_data(JSON_FILE)
    
    # Dictionary to store errors by subject and keypoint
    errors_by_subject_keypoint = defaultdict(lambda: defaultdict(list))
    
    # Dictionary to store all errors by keypoint (for ALL column)
    all_errors_by_keypoint = defaultdict(list)
    
    # Set to keep track of all keypoint names
    keypoint_names = set()
    
    # Process each image
    for item in data:
        image_name = item['image_name']
        
        # Extract subject name from image name
        subject_name = image_name.split('_img')[0]
        
        # Process keypoints
        for kp in item['keypoints']:
            kp_name = kp['keypoint_name']
            distance = kp['distance']
            
            # Add to subject-specific errors
            errors_by_subject_keypoint[subject_name][kp_name].append(distance)
            
            # Add to all errors
            all_errors_by_keypoint[kp_name].append(distance)
            
            # Add keypoint name to set
            keypoint_names.add(kp_name)
    
    # Convert keypoint names set to sorted list
    keypoint_names = sorted(list(keypoint_names))
    
    # Prepare data for DataFrame
    subjects = sorted(errors_by_subject_keypoint.keys())
    
    # Create DataFrame columns: ALL, followed by individual keypoints
    columns = keypoint_names + ['ALL']
    
    # Initialize DataFrame for mean ± std
    df = pd.DataFrame(index=subjects + ['Total'], columns=columns)
    
    # Initialize DataFrames for mean and std separately (for heatmap)
    df_mean = pd.DataFrame(index=subjects + ['Total'], columns=columns)
    df_std = pd.DataFrame(index=subjects + ['Total'], columns=columns)
    
    # Initialize DataFrame for raw data (for boxplot)
    raw_data = []
    
    # Fill DataFrames
    for subject in subjects:
        # Calculate ALL column (all keypoints for this subject)
        all_errors = []
        for kp_name in keypoint_names:
            all_errors.extend(errors_by_subject_keypoint[subject][kp_name])
            
            # Add to raw data
            for error in errors_by_subject_keypoint[subject][kp_name]:
                raw_data.append({
                    'Subject': subject,
                    'Keypoint': kp_name,
                    'Error': error
                })
        
        mean_all = np.mean(all_errors) if all_errors else 0
        std_all = np.std(all_errors) if all_errors else 0
        
        df.at[subject, 'ALL'] = f"{mean_all:.2f} ± {std_all:.2f}"
        df_mean.at[subject, 'ALL'] = mean_all
        df_std.at[subject, 'ALL'] = std_all
        
        # Calculate for each keypoint
        for kp_name in keypoint_names:
            errors = errors_by_subject_keypoint[subject][kp_name]
            if errors:
                mean_kp = np.mean(errors)
                std_kp = np.std(errors)
                df.at[subject, kp_name] = f"{mean_kp:.2f} ± {std_kp:.2f}"
                df_mean.at[subject, kp_name] = mean_kp
                df_std.at[subject, kp_name] = std_kp
            else:
                df.at[subject, kp_name] = "N/A"
                df_mean.at[subject, kp_name] = np.nan
                df_std.at[subject, kp_name] = np.nan
    
    # Calculate Total row
    # ALL column (all subjects, all keypoints)
    all_errors = []
    for kp_name in keypoint_names:
        all_errors.extend(all_errors_by_keypoint[kp_name])
        
        # Add to raw data
        for error in all_errors_by_keypoint[kp_name]:
            raw_data.append({
                'Subject': 'Total',
                'Keypoint': kp_name,
                'Error': error
            })
    
    mean_all = np.mean(all_errors) if all_errors else 0
    std_all = np.std(all_errors) if all_errors else 0
    
    df.at['Total', 'ALL'] = f"{mean_all:.2f} ± {std_all:.2f}"
    df_mean.at['Total', 'ALL'] = mean_all
    df_std.at['Total', 'ALL'] = std_all
    
    # Each keypoint column (all subjects)
    for kp_name in keypoint_names:
        errors = all_errors_by_keypoint[kp_name]
        if errors:
            mean_kp = np.mean(errors)
            std_kp = np.std(errors)
            df.at['Total', kp_name] = f"{mean_kp:.2f} ± {std_kp:.2f}"
            df_mean.at['Total', kp_name] = mean_kp
            df_std.at['Total', kp_name] = std_kp
        else:
            df.at['Total', kp_name] = "N/A"
            df_mean.at['Total', kp_name] = np.nan
            df_std.at['Total', kp_name] = np.nan
    
    # Convert raw data to DataFrame
    df_raw = pd.DataFrame(raw_data)
    
    # Save DataFrame to CSV
    csv_path = os.path.join(OUTPUT_DIR, OUTPUT_CSV)
    df.to_csv(csv_path)
    print(f"Analysis saved to {csv_path}")
    
    # Create a visual table using matplotlib
    plt.figure(figsize=(12, len(subjects) + 3))
    
    # Hide axes
    ax = plt.subplot(111)
    ax.axis('off')
    
    # Create table
    table = ax.table(
        cellText=df.values,
        rowLabels=df.index,
        colLabels=df.columns,
        cellLoc='center',
        loc='center'
    )
    
    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Add title
    plt.title('Keypoint Error Analysis (Mean ± Std in pixels)', fontsize=16, pad=20)
    
    # Save figure
    plot_path = os.path.join(OUTPUT_DIR, OUTPUT_PLOT)
    plt.savefig(plot_path, bbox_inches='tight', dpi=150)
    print(f"Plot saved to {plot_path}")
    
    # Create enhanced heatmap with mean and std
    plt.figure(figsize=(14, len(subjects) + 3))
    
    # Create the base heatmap with mean values
    ax = sns.heatmap(df_mean.astype(float), annot=False, cmap="YlOrRd", 
                linewidths=.5, cbar_kws={'label': 'Mean Error (pixels)'})
    
    # Add text annotations with mean ± std
    for i in range(len(df_mean.index)):
        for j in range(len(df_mean.columns)):
            mean_val = df_mean.iloc[i, j]
            std_val = df_std.iloc[i, j]
            
            if not np.isnan(mean_val) and not np.isnan(std_val):
                text = f"{mean_val:.2f}\n±{std_val:.2f}"
                text_color = "white" if mean_val > 600 else "black"  # Adjust threshold as needed
                
                # Add text annotation
                ax.text(j + 0.5, i + 0.5, text, 
                        ha="center", va="center", color=text_color,
                        fontsize=9)
    
    plt.title('Keypoint Error Analysis (Mean ± Std in pixels)', fontsize=16)
    plt.tight_layout()
    
    # Save enhanced heatmap
    heatmap_path = os.path.join(OUTPUT_DIR, OUTPUT_HEATMAP)
    plt.savefig(heatmap_path, bbox_inches='tight', dpi=150)
    print(f"Enhanced heatmap saved to {heatmap_path}")
    
    # Create boxplot of errors by keypoint
    plt.figure(figsize=(16, 8))
    sns.boxplot(x='Keypoint', y='Error', data=df_raw)
    plt.title('Distribution of Keypoint Errors', fontsize=16)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save boxplot
    boxplot_path = os.path.join(OUTPUT_DIR, OUTPUT_BOXPLOT)
    plt.savefig(boxplot_path, bbox_inches='tight', dpi=150)
    print(f"Boxplot saved to {boxplot_path}")
    
    # Print summary
    print("\nSummary of Keypoint Error Analysis:")
    print(f"Total subjects: {len(subjects)}")
    print(f"Total keypoints: {len(keypoint_names)}")
    print(f"Overall mean error: {mean_all:.2f} pixels")
    print(f"Overall std error: {std_all:.2f} pixels")
    
    # Return DataFrame for further analysis if needed
    return df

if __name__ == "__main__":
    analyze_keypoint_errors()