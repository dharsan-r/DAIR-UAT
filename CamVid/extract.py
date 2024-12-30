import numpy as np
import cv2
import os
import csv
from tqdm import tqdm

# Open CSV file to get the mapping between label names and their RGB values
rgb_dict = {}
print("Loading class dictionary...")
with open('class_dict.csv', 'r') as file:
    reader = csv.reader(file)
    # Skip the first row (header)
    next(reader)  # This skips the first row
    # Read the rest of the rows
    for row in reader:
        label_name = row[0]
        r, g, b = int(row[1]), int(row[2]), int(row[3])
        rgb_dict[label_name] = [r, g, b]  # Mapping RGB values to label name

print(f"Found {len(rgb_dict)} classes in dictionary")

# Path to the root directory where masks will be saved
root_path = "test_masks_extracted"
# Path where labeled PNG files are located
read_path = "test_labels"

# comment above and unccoment this to do train labels
# root_path = "train_final"
# read_path = "train_labels"

# Get total number of PNG files
png_files = [f for f in os.listdir(read_path) if f.endswith('.png')]
total_files = len(png_files)

print(f"\nProcessing {total_files} images...")

# Iterate through each PNG file in the 'test_labels' folder
for file_name in tqdm(png_files, desc="Processing images"):
    # Read the mask image (it should be a color image with RGB labels)
    image_path = os.path.join(read_path, file_name)
    img = cv2.imread(image_path)
    
    # Get the base name of the file (without extension)
    base_name = os.path.splitext(file_name)[0]
    
    # Create a directory for the current image in the output folder (if not exists)
    output_folder = os.path.join(root_path, base_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    # Iterate through each RGB value and its label
    for label_name, [r, g, b] in rgb_dict.items():
        # Create a boolean mask where the RGB values match
        mask = np.all(img == [b, g, r], axis=2)  # OpenCV uses BGR format
        
        # Only create and save the mask if the label exists in the image
        if np.any(mask):  # Check if there are any True values in the mask
            # Convert boolean mask to uint8 (0 and 255)
            binary_mask = mask.astype(np.uint8) * 255
            
            # Save the binary mask
            output_path = os.path.join(output_folder, f"{label_name}.png")
            cv2.imwrite(output_path, binary_mask)

print("\nDone! Masks have been extracted and saved.")