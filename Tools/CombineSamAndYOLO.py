import ultralytics
from IPython.display import display, Image
from roboflow import Roboflow
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
import glob
from segment_anything import sam_model_registry, SamPredictor
import torch
device= "cuda"
ultralytics.checks()

from roboflow import Roboflow
rf = Roboflow(api_key="atzoRMP07uwlakjX3XXY")
#project object
project = rf.workspace("vkr-v2").project("vkrrr")
dataset = project.version(5).download("yolov8")

model = YOLO('yolov8n.pt')


print("Torch version:", torch.__version__)

print("Is CUDA enabled?", torch.cuda.is_available())





#functions to show masks and points and boxes

def show_mask(mask, ax,  class_name, random_color=False):
    if random_color:
        #use random color for the masks
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        #use blue color
        #the fourth value stands for opacity
        if class_name == "person":
            # Corresponds to Pedestrian: 64, 64, 0
            color = np.array([64 / 255, 64 / 255, 0 / 255, 1])
        elif class_name == "car":
            # Car: 64, 0, 128
            color = np.array([64 / 255, 0 / 255, 128 / 255, 1])
        elif class_name == "bus":
            # Use Truck_Bus for bus: 192, 128, 192
            color = np.array([192 / 255, 128 / 255, 192 / 255, 1])
        elif class_name == "traffic light":
            # TrafficLight: 0, 64, 64
            color = np.array([0 / 255, 64 / 255, 64 / 255, 1])
        elif class_name == "bicycle":
            # Bicyclist: 0, 128, 192
            color = np.array([0 / 255, 128 / 255, 192 / 255, 1])
        elif class_name == "truck":
            # Use Truck_Bus for truck: 192, 128, 192
            color = np.array([192 / 255, 128 / 255, 192 / 255, 1])

        elif class_name == "train":
            # Train: 192, 64, 128
            color = np.array([192 / 255, 64 / 255, 128 / 255, 1])

        elif class_name == "motorcycle":
            # MotorcycleScooter: 192, 0, 192
            color = np.array([192 / 255, 0 / 255, 192 / 255, 1])
        else:
            # Default color for any unexpected class
            color = np.array([30 / 255, 144 / 255, 255 / 255, 1])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):

    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))


#functions to process images with YOLO and drawing the bound boxes











def predict(chosen_model, img, classes=[], conf=0.5):
    if classes:
        results= chosen_model.predict(img,classes=classes,conf=conf)
    else:
        results= chosen_model.predict(img,conf=conf)
    return results

def predict_and_detect(chosen_mode, img, classes=[], conf=0.5):
    results = predict(chosen_mode, img, classes=classes, conf=conf)
    #taggle it on or off to draw boxes
    for result in results:
        for box in result.boxes:
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), 2)

            cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
    return img, results

def process_all_in_dir(dir, option):






    input_folder = dir
    output_folder = r"C:\Users\jason\OneDrive\Desktop\QMind\DAIR-UAT\Processed_Results"  # folder to save processed images

    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get list of all .png image files in the input folder
    image_files = glob.glob(os.path.join(input_folder, "*.png"))
    sam_checkpoint = "./sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)


    for image_path in image_files:
        # Read the image

        imageSam = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        #Use sam to do the prediction on the image
        predictor.set_image(imageSam)


        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read {image_path}. Skipping...")
            continue

        # Process image
        modified_img, results = predict_and_detect(model, image, classes=[], conf=0.5)

        if(option==0):
            filename = os.path.basename(image_path)
            save_path = os.path.join(output_folder, filename)
            cv2.imwrite(save_path, modified_img)

            # Optional: Display the image (commented out by default)
            cv2.imshow("Image", modified_img)
            cv2.waitKey(1)  # small delay to display each image; adjust as needed
            # Save the processed image to output folder
        elif(option==1):
            #only 1 image so actually only 1 result in results
            i=0
            for result in results:
                boxes = result.boxes

            #save file in another dir, now specify the path of the saved image
            filename = os.path.basename(image_path)
            base,ext = os.path.splitext(filename)
            save_image_path =  f"C:/Users/jason/OneDrive/Desktop/QMind/DAIR-UAT/YOLO V8 Classification of Masks/Images and masks/Advanced Masks with Windows/{base}_converted{ext}"

            #height width of the image
            height, width = imageSam.shape[:2]
            dpi=100
            width_in = width/dpi
            height_in = height/dpi
            bbox = boxes.xyxy.tolist()
            yolo_classes = [
                "person",
                "car",
                "bus",
                "traffic light",
                "bicycle",
                "truck",

                "train",

                "motorcycle"
            ]
            #we want masks for every box
            if(len(bbox)>0):
                print(bbox)
                multi_masks = []
                img_classes = []
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    input_box = np.array([x1, y1, x2, y2])


                    #this gives us the classification index of the box
                    class_index = int(box.cls[0])
                    class_name = results[0].names[class_index]

                    print("Class Index:", class_index, "Class Name:", class_name)
                    if(class_name in yolo_classes):
                    #now masks[0] is the mask for the box
                        masks, _, _ = predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=input_box[None, :],
                        multimask_output=False,
                        )
                        multi_masks.append(masks[0])
                        img_classes.append(class_name)



            #now we have multiple masks

                # bbox was an array of boxes, now just one box

            # bbox = bbox[0]
            # input_box = np.array(bbox)
            # masks, _, _ = predictor.predict(
            #     point_coords=None,
            #     point_labels=None,
            #     box=input_box[None, :],
            #     multimask_output=False,
            # )


            plt.figure(figsize=(width_in, height_in), dpi=dpi)
            plt.imshow(image)
            count=0
            print(img_classes)
            #everytime the classname passed in will be different
            for mask in multi_masks:
                show_mask(mask, plt.gca(), img_classes[count])
                count+=1


            show_box(input_box, plt.gca())
            plt.axis('off')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            plt.savefig(save_image_path)
            plt.show()


    # If using imshow in a loop, call cv2.destroyAllWindows() after loop ends
    cv2.destroyAllWindows()

    print("Processing complete!")

dir =r"C:\Users\jason\OneDrive\Desktop\QMind\DAIR-UAT\YOLO V8 Classification of Masks\Images and masks\Raw Images"
#option 0 to display images with bounding boxes, option 1 to get the resulting boxes
# process_all_in_dir(dir,0)
process_all_in_dir(dir,1)