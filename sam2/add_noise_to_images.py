import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

def noisify_image(image:np.array, noise:str="random", threshold=0.1):
    # add noise to image
    # snow is generally greyscale, so noise values should be in that range
        # uint8 dtype limits cv2 randn to [0, 255]
    noise_mask = np.zeros(shape=(image.shape[0], image.shape[1], 3), dtype=np.uint8)

    if noise == "gaussian":
        # apply gaussian noise
        cv2.randn(noise_mask, mean=(128, 128, 128), stddev=(40, 40, 40))
        noise_mask = (noise_mask * 0.5).astype(np.uint8) # dilute the noise so that its application to the iamge is more realistic
        image = np.add(image, noise_mask)

    # add more noises methods here...
    elif noise == "random":
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if np.random.random() <= threshold:
                    image[i][j] = (np.random.rand(3) * 255).astype(np.uint32)
        

    return image

# Example usage
if __name__ == "__main__":
    image=Image.open("../0001TP_009240.png") # a sample CamVid image outside the SAM folder
    image=np.array(image.convert("RGB"))

    print("Before Noise:")
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

    # thresholds 0.3 and above are a bit brutal - like a snow day snowstorm.
    image = noisify_image(image, noise="random", threshold=0.1)

    print("After Noise:")
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    plt.axis('off')
    plt.show()