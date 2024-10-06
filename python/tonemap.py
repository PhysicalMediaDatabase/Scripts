import io
import os
import cv2
import colour
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.path import Path
from skimage.transform import downscale_local_mean
from common import eotf_ST2084, eotf_inverse_ST2084


lut = None


def tonemap_initialize():
    global lut
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lut_file_path = os.path.join(script_dir, 'HDR-SDR+LM.cube')
    lut = colour.io.read_LUT(lut_file_path)
    return


# Define empty function signature
def tonemap_image(input_image, output_image):
    
    # Load image data ---------------------------------------------------------

    # Read image file into memory
    image_data = Image.open(input_image)

    # Scale image data from 0-255 to 0-1
    # TODO - Why? Isn't the data encoded in 10-16bit?
    image_data = np.array(image_data) / 255.0

    # Load a 3D LUT and apply it to the image data ---------------------------

    # Load the LUT into memory from the file
    if lut is None:
        tonemap_initialize()

    # Apply the LUT to the image data
    image_lut_applied = lut.apply(image_data)

    # Scale the pixel values back to the range [0, 255] and convert them to 8-bit integers for saving
    image_data_wlut = (image_lut_applied * 255).astype(np.uint8)

    # Save the image --------------------------------------------------------

    # Open the original image data as an image
    original_image = Image.fromarray(image_data_wlut)

    # Convert the original image to a NumPy array in RGB color space
    original_image_cv = np.array(original_image.convert('RGB'))

    # Save the image to the output path
    cv2.imwrite(output_image, cv2.cvtColor(original_image_cv, cv2.COLOR_RGB2BGR))

    return