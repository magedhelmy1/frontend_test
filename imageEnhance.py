import cv2
import numpy as np
from skimage.exposure import histogram
from PIL import Image


def enhance_image(image):
    # Calculate intensity histogram
    hist, hist_centers = histogram(image)

    # Find min intensity
    percentage_to_drop = 0.01
    nr_of_pixels = image.shape[0] * image.shape[1]
    min_intensity = 0
    max_intensity = 255
    for i in range(1, len(hist)):  # Skip zero intensity pixels
        if hist[i] > nr_of_pixels * percentage_to_drop:
            min_intensity = hist_centers[i]
            break

    # Find end intensity
    for i in range(len(hist) - 1, 0, -1):
        if hist[i] > nr_of_pixels * percentage_to_drop:
            max_intensity = hist_centers[i]
            break

    image = image.astype(np.float32)
    image = np.clip(image, min_intensity, max_intensity)
    image = (image - min_intensity) / (max_intensity - min_intensity)

    return image


class ImageEnhancement(object):
    def __init__(self):
        pass

    def __call__(self, image):
        init_dtype = image.dtype

        enhanced_image = enhance_image(image)

        if init_dtype == np.uint8:
            enhanced_image = (enhanced_image * 255.0).astype(np.uint8)

        return enhanced_image


img_temp = np.asarray(Image.open("1.png"))
image_enhancement = ImageEnhancement()
org_enhanced = image_enhancement(img_temp)

cv2.imshow("before_enh", img_temp)
cv2.imshow("after_enh", org_enhanced)
cv2.waitKey(0)
cv2.destroyAllWindows()
