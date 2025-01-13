import cv2
from ultralytics import YOLO
import os
import glob
model = YOLO("yolov8n.pt")

def predict(chosen_model, img, classes=[], conf=0.5):
    if classes:
        results= chosen_model.predict(img,classes=classes,conf=conf)
    else:
        results= chosen_model.predict(img,conf=conf)
    return results

def predict_and_detect(chosen_mode, img, classes=[], conf=0.5):
    results = predict(chosen_mode, img, classes=classes, conf=conf)

    for result in results:
        for box in result.boxes:
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), 2)

            cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
    return img, results


if "__main__" == __name__:
    input_folder = r"C:\Users\jason\OneDrive\Desktop\QMind\DAIR-UAT\YOLO V8 Classification of Masks\Images and masks\Masks"
    output_folder = r"C:\Users\jason\OneDrive\Desktop\QMind\DAIR-UAT\Processed_Results"  # folder to save processed images

    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get list of all .png image files in the input folder
    image_files = glob.glob(os.path.join(input_folder, "*.png"))

    for image_path in image_files:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read {image_path}. Skipping...")
            continue

        # Process image
        modified_img, results = predict_and_detect(model, image, classes=[], conf=0.5)

        # Save the processed image to output folder
        filename = os.path.basename(image_path)
        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, modified_img)

        # Optional: Display the image (commented out by default)
        cv2.imshow("Image", modified_img)
        cv2.waitKey(1)  # small delay to display each image; adjust as needed

    # If using imshow in a loop, call cv2.destroyAllWindows() after loop ends
    cv2.destroyAllWindows()

    print("Processing complete!")
