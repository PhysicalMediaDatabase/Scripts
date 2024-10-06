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


lut = None  # Undefined global variable, must call initialize function


def wcg_initialize():
    global lut
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lut_file_path = os.path.join(script_dir, 'HDR-SDR+LM.cube')
    lut = colour.io.read_LUT(lut_file_path)
    return


# Define empty function signature
def wcg_visualize(input_image, output_image):

    # Load image data ---------------------------------------------------------

    # Read image file into memory
    image_data = Image.open(input_image)

    # Scale image data from 0-255 to 0-1
    # TODO - Why? Isn't the data encoded in 10-16bit?
    image_data = np.array(image_data) / 255.0

    # Convert image data to linear luminance values using PQ EOTF
    image_linear = eotf_ST2084(image_data)

    # Create downsampled image at 1/5th resolution
    # This must be to reduce pixels/processing for scatter plot
    image_linear_scatter = downscale_local_mean(image_linear, (5, 5, 1))

    # Convert linear RGB to XYZ and xy ---------------------------------------------------

    # Convert linear light values to XYZ color space
    xyz_data = colour.RGB_to_XYZ(image_linear,
        colour.RGB_COLOURSPACES['ITU-R BT.2020'].whitepoint,
        colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65'],
        colour.RGB_COLOURSPACES['ITU-R BT.2020'].matrix_RGB_to_XYZ)
    
    # Convert XYZ color space to xy color space
    xy_data = colour.XYZ_to_xy(xyz_data)

    # Chromacity coordinates of the colour spaces -----------------------------

    # Get the chromacity coordinates of the Rec. 2020 color space
    rec_2020_xy = colour.RGB_COLOURSPACES['ITU-R BT.2020'].primaries

    # Get the chromacity coordinates of the Rec. 709 color space
    rec_709_xy = colour.RGB_COLOURSPACES['ITU-R BT.709'].primaries

    # Get the chromacity coordinates of the DCI-P3 color space
    p3_d65_xy = colour.RGB_COLOURSPACES['DCI-P3'].primaries

    # Create a mask for Rec. 709 colors ---------------------------------------

    # Define a path in the xy chromaticity diagram representing the Rec. 709 color space
    rec_709_path = Path(rec_709_xy)

    # Create a boolean mask indicating whether each point in xy_data is inside the Rec. 709 path
    # xy_data is reshaped to a 2D array for compatibility with the contains_points method
    mask = rec_709_path.contains_points(xy_data.reshape(-1, 2))

    # Reshape the mask to match the original image dimensions (height and width)
    mask = mask.reshape(xy_data.shape[:2])

    # Convert Rec. 709 colors to black and white ------------------------------

    # Calculate the grayscale equivalent of the colors in the image that are inside the Rec. 709 color space
    rgb_grey = np.dot(image_linear[mask], [0.2989, 0.5870, 0.1140])

    # Replace the colors in the image that are inside the Rec. 709 color space with their grayscale equivalents
    image_linear[mask] = rgb_grey[:, None]

    # Apply the inverse ST 2084 EOTF to the image to convert the pixel values back to a non-linear gamma space
    image_data_output = eotf_inverse_ST2084(image_linear)

    # Load a 3D LUT and apply it to the image data ---------------------------

    # Load the LUT into memory from the file
    if lut is None:
        wcg_initialize()

    # Apply the LUT to the image data
    image_lut_applied = lut.apply(image_data_output)

    # Scale the pixel values back to the range [0, 255] and convert them to 8-bit integers for saving
    image_data_wlut = (image_lut_applied * 255).astype(np.uint8)

    # Convert downsampled RGB to XYZ and xy for scatter plot ------------------

    # Convert the downsampled linear light values to XYZ color space
    xyz_data_scatter = colour.RGB_to_XYZ(image_linear_scatter,
        colour.RGB_COLOURSPACES['ITU-R BT.2020'].whitepoint,
        colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65'],
        colour.RGB_COLOURSPACES['ITU-R BT.2020'].matrix_RGB_to_XYZ)
    
    # Convert the downsampled XYZ color space values to xy color space
    xy_data_scatter = colour.XYZ_to_xy(xyz_data_scatter)

    # Create a scatter plot overlay -------------------------------------------

    # Create a new figure and subplot with specified size and resolution
    # (8inches x 8inches, with 400 dots per inch)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=400)

    # Set the figure's background color to be transparent
    fig.patch.set_facecolor('none')

    # Add a filled polygon representing the Rec. 2020 color space to the plot
    ax.add_patch(plt.Polygon(rec_2020_xy, fill=True, color='#2e2e2e', alpha=1))

    # Convert the xy chromaticity data to XYZ color space
    XYZ_points_scatter = colour.xy_to_XYZ(xy_data_scatter)

    # Convert the XYZ color space data to sRGB color space
    sRGB_points_scatter = colour.XYZ_to_sRGB(XYZ_points_scatter)

    # Clip the sRGB color space data to the range [0, 1]
    sRGB_points_scatter = np.clip(sRGB_points_scatter, 0, 1)

    # Define some keyword arguments for the scatter plot
    scatter_kwargs = {"s": 0.015, "alpha": 0.05}

    # Create a scatter plot of the xy chromaticity data with the sRGB color space data as the color
    ax.scatter(xy_data_scatter[..., 0], xy_data_scatter[..., 1], c=sRGB_points_scatter.reshape(-1, 3), **scatter_kwargs)

    # Add an outline of the Rec. 2020 color space
    ax.add_patch(plt.Polygon(rec_2020_xy, fill=False, edgecolor='#000000', lw=4))

    # Add an outline of the Rec. 709 color space
    ax.add_patch(plt.Polygon(rec_709_xy, fill=False, edgecolor='#ffffff', lw=1))

    # Add an outline of the P3-D65 color space
    ax.add_patch(plt.Polygon(p3_d65_xy, fill=False, edgecolor='#ffffff', lw=1))

    # Set the x and y axis limits to [0, 1] and remove the ticks
    ax.set(xticks=[], yticks=[], xlim=[0, 1], ylim=[0, 1])

    # Turn off the axis lines and labels
    ax.axis('off')

    # Save the scatter plot overlay -------------------------------------------

    # Create an in-memory buffer to save the plot to
    buf = io.BytesIO()

    # Save the plot to the buffer as a PNG image with tight bounding box and no padding
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)

    # Close the plot to free up memory
    plt.close(fig)

    # Reset the buffer position to the beginning
    buf.seek(0)

    # Open the buffer as an image
    overlay_image = Image.open(buf)

    # Open the original image data as an image
    original_image = Image.fromarray(image_data_wlut)

    # Resize the overlay image to 28% of the original image width -------------

    # Calculate the width of the overlay image using a scale factor
    overlay_width = int(original_image.width * 0.28)

    # Calculate the height based on the aspect ratio of the overlay image
    overlay_height = int(overlay_width * overlay_image.height / overlay_image.width)

    # Resize the overlay image to the calculated width and height
    overlay_image = overlay_image.resize((overlay_width, overlay_height))

    # Calculate the position to paste the overlay image onto the original image
    overlay_position = (0, original_image.height - overlay_height)

    # Paste the overlay image onto the original image at the calculated position
    original_image.paste(overlay_image, overlay_position, overlay_image)

    # Convert the original image to a NumPy array in RGB color space
    original_image_cv = np.array(original_image.convert('RGB'))

    # Define labels and font properties ---------------------------------------

    # Define the labels for the color spaces
    labels = ["Rec. 2020", "DCI-P3", "Rec. 709"]

    # Define the font properties for the labels
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Define the font scale for the labels
    font_scale = 0.9

    # Define the font thickness for the labels
    font_thickness = 2

    # Calculate the width and height of the labels --------------------------------

    # Calculate the width of a space character in the font
    spacing_width = cv2.getTextSize(" ", font, font_scale, font_thickness)[0][0]

    # Calculate the total width of the labels
    total_text_width = sum([cv2.getTextSize(label, font, font_scale, font_thickness)[0][0] for label in labels]) + (len(labels) - 1) * spacing_width

    # Calculate the maximum height of the labels
    max_text_height = max([cv2.getTextSize(label, font, font_scale, font_thickness)[0][1] for label in labels])

    # Determine the starting position for the text ----------------------------

    # Calculate the x position to start the text
    x_text_start = int(original_image_cv.shape[1] * 0.05)

    # Calculate the y position to start the text
    y_text_start = original_image_cv.shape[0] - max_text_height - int(original_image_cv.shape[0] * 0.01)

    # Calculate the width and height of the rectangle -------------------------

    # Calculate the width of the rectangle
    rectangle_width = int(total_text_width * 1.03)

    # Calculate the height of the rectangle
    rectangle_height = int(max_text_height * 1.5)

    # Compute rectangle position so as to be centered -------------------------

    # Calculate the x position of the rectangle
    rectangle_x_pos = x_text_start - (rectangle_width - total_text_width) // 2

    # Calculate the y position of the rectangle
    rectangle_y_pos = y_text_start - (rectangle_height - max_text_height) // 2

    # Draw the rectangle ------------------------------------------------------

    # Draw a filled rectangle on the image
    cv2.rectangle(original_image_cv, (rectangle_x_pos, rectangle_y_pos), (rectangle_x_pos + rectangle_width, rectangle_y_pos + rectangle_height), (0, 0, 0), thickness=-1)

    # Draw the labels --------------------------------------------------------

    # Calculate the starting x position for the labels
    x_pos = x_text_start

    # Calculate the starting y position for the labels
    y_pos = y_text_start + max_text_height

    # Draw each label on the image
    for label in labels:
        # Draw the label on the image
        cv2.putText(original_image_cv, label, (x_pos, y_pos), font, font_scale, (255, 255, 255), thickness=font_thickness, lineType=cv2.LINE_AA)

        # Calculate the width of the label
        text_width, _ = cv2.getTextSize(label, font, font_scale, font_thickness)[0]

        # Update the x position for the next label
        x_pos += text_width + spacing_width

    # Save the images --------------------------------------------------------

    # Save the image to the output path
    cv2.imwrite(output_image, cv2.cvtColor(original_image_cv, cv2.COLOR_RGB2BGR))

    return