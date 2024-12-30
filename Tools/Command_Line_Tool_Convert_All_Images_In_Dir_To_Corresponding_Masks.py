import cv2
import torch
from PIL import Image, ImageDraw
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from matplotlib import pyplot as plt
import numpy as np
import supervision as sv
import sys
import os
import os
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_b"
sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
mask_generator = SamAutomaticMaskGenerator(sam)
sam.to(device=DEVICE)



# usage: Change mask_path_strip to the directory you want to store masks on your computer
# Run the script through command line, the first argument is the directory where all your images are stored
# Make sure all dependencies of SAM and this script is installed

def get_png_paths(directory):
    """
    Takes a directory path like 'c:/images' and returns a list of PNG filenames
    without the directory path, like ['1.png', 'dsk.png', ...]

    Args:
        directory (str): Path to the directory to search

    Returns:
        list: List of PNG filenames
    """
    # Get list of all files and filter for PNGs
    return [f for f in os.listdir(directory) if f.lower().endswith('.png')]


#raw image dir is in the format like C:/user/images
#png_paths is in the format like aswd.png
def save_masks_given_raw_images_path(raw_images_dir, png_path):
    image_name = png_path

    image_path_strip = raw_images_dir
    image_path = image_path_strip + image_name

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = mask_generator.generate(image_rgb)

    array = image_name.split('.')
    #you must change the path here to the path on your own machine!!!!!
    mask_path_strip = "C:/Users/jason/Sam1/segment-anything/CamVid/Images and masks/Masks/"
    mask_path = mask_path_strip + array[0] + "_converted.png"

    mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)  # You can choose other color maps as well
    detections = sv.Detections.from_sam(mask)

    # Annotate the image
    # create a blank image and annotate the mask onto the blank image and the original image
    height, width, channels = image_rgb.shape
    blank_image = np.zeros((height, width, 3), dtype=np.uint8)

    annotated_image_rgb = mask_annotator.annotate(image_rgb, detections)
    mask_only = mask_annotator.annotate(blank_image, detections)
    # Convert annotated image back to BGR for OpenCV
    annotated_image_bgr = cv2.cvtColor(annotated_image_rgb, cv2.COLOR_RGB2BGR)

    # saving the file

    # showing the annotated image(mask over original image)
    print(mask_path+"is completed")

    cv2.imwrite(mask_path, mask_only)




raw_image_dir = sys.argv[1]
png_paths = get_png_paths(raw_image_dir)
for png_path in png_paths:
    save_masks_given_raw_images_path(raw_image_dir, png_path)

#Creating an image object



# def show_output(result_dict,axes=None):
#      if axes:
#         ax = axes
#      else:
#         ax = plt.gca()
#         ax.set_autoscale_on(False)
#      sorted_result = sorted(result_dict, key=(lambda x: x['area']),      reverse=True)
#      # Plot for each segment area
#      for val in sorted_result:
#         mask = val['segmentation']
#         img = np.ones((mask.shape[0], mask.shape[1], 3))
#         color_mask = np.random.random((1, 3)).tolist()[0]
#         for i in range(3):
#             img[:,:,i] = color_mask[i]
#             ax.imshow(np.dstack((img, mask*0.5)))



# _,axes = plt.subplots(1,2, figsize=(16,16))
# axes[0].imshow(image_rgb)
# show_output(mask, axes[1])
# plt.show()


