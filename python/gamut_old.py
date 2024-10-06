import os
import io
import cv2
import colour
from colour.utilities import ColourUsageWarning
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path
from skimage.transform import downscale_local_mean
import argparse
import warnings
import gc

warnings.filterwarnings("ignore", category=ColourUsageWarning)

def eotf_ST2084(N, m1=0.1593017578125, m2=78.84375, c1=0.8359375, c2=18.8515625, c3=18.6875):
    L = (np.clip(np.power(N, (1/m2)) - c1, 0, None) / (c2 - c3 * np.power(N, (1/m2))))
    L = np.power(L, 1/m1)
    return L

def eotf_inverse_ST2084(N, m1=0.1593017578125, m2=78.84375, c1=0.8359375, c2=18.8515625, c3=18.6875):
    Y_p = np.power(N, m1)
    C = np.power((c1 + c2 * Y_p) / (c3 * Y_p + 1), m2)
    return C

# Create the parser
parser = argparse.ArgumentParser(description='Process images in a folder.')

# Add the arguments
parser.add_argument('input_folder', type=str, help='The folder containing the images to process')
parser.add_argument('output_folder', type=str, help='The folder where the processed images will be saved')

# Parse the arguments
args = parser.parse_args()

# Use the arguments in the script
image_directory = args.input_folder
output_directory = args.output_folder

# Replace the line where image_files is defined with the following:
image_files = [os.path.join(image_directory, f) for f in os.listdir(image_directory) if f.endswith(('.jpg', '.jpeg', '.png'))]

if not image_files:
    print("No images found in the input folder. Exiting.")
    raise SystemExit
num = 0
for image_file in image_files:
    image_data = Image.open(image_file)
    image_data = np.array(image_data) / 255.0
    image_linear = eotf_ST2084(image_data)
    image_linear_scatter = downscale_local_mean(image_linear, (5, 5, 1))

    # Convert to XYZ and xy
    xyz_data = colour.RGB_to_XYZ(image_linear,
    colour.RGB_COLOURSPACES['ITU-R BT.2020'].whitepoint,
    colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65'],
    colour.RGB_COLOURSPACES['ITU-R BT.2020'].matrix_RGB_to_XYZ)
    xy_data = colour.XYZ_to_xy(xyz_data)

    # Chromaticity coordinates of the colour spaces
    rec_2020_xy = colour.RGB_COLOURSPACES['ITU-R BT.2020'].primaries
    rec_709_xy = colour.RGB_COLOURSPACES['ITU-R BT.709'].primaries
    p3_d65_xy = colour.RGB_COLOURSPACES['DCI-P3'].primaries

    # Create a mask for Rec. 709 colours
    rec_709_path = Path(rec_709_xy)
    mask = rec_709_path.contains_points(xy_data.reshape(-1, 2))
    mask = mask.reshape(xy_data.shape[:2])

    # Convert Rec. 709 colours to B&W
    rgb_grey = np.dot(image_linear[mask], [0.2989, 0.5870, 0.1140])
    image_linear[mask] = rgb_grey[:, None]
    image_data_output = eotf_inverse_ST2084(image_linear)

    # Load LUT and apply to the image data
    lut = colour.io.read_LUT('HDR-SDR+LM.cube')
    image_lut_applied = lut.apply(image_data_output)
    image_data_wlut = (image_lut_applied * 255).astype(np.uint8)

    # Delete variables
    del image_data, image_linear, image_data_output, image_lut_applied
    gc.collect()

    # Convert to XYZ and xy for scatter plot
    xyz_data_scatter = colour.RGB_to_XYZ(image_linear_scatter,
    colour.RGB_COLOURSPACES['ITU-R BT.2020'].whitepoint,
    colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65'],
    colour.RGB_COLOURSPACES['ITU-R BT.2020'].matrix_RGB_to_XYZ)
    xy_data_scatter = colour.XYZ_to_xy(xyz_data_scatter)

    # Create the scatter plot overlay
    fig, ax = plt.subplots(figsize=(8, 8), dpi=400)
    fig.patch.set_facecolor('none')
    ax.add_patch(plt.Polygon(rec_2020_xy, fill=True, color='#2e2e2e', alpha=1))
    XYZ_points_scatter = colour.xy_to_XYZ(xy_data_scatter)
    sRGB_points_scatter = colour.XYZ_to_sRGB(XYZ_points_scatter)
    sRGB_points_scatter = np.clip(sRGB_points_scatter, 0, 1)
    scatter_kwargs = {"s": 0.015, "alpha": 0.05}
    ax.scatter(xy_data_scatter[..., 0], xy_data_scatter[..., 1], c=sRGB_points_scatter.reshape(-1, 3), **scatter_kwargs)
    ax.add_patch(plt.Polygon(rec_2020_xy, fill=False, edgecolor='#000000', lw=4))
    ax.add_patch(plt.Polygon(rec_709_xy, fill=False, edgecolor='#ffffff', lw=1))
    ax.add_patch(plt.Polygon(p3_d65_xy, fill=False, edgecolor='#ffffff', lw=1))
    ax.set(xticks=[], yticks=[], xlim=[0, 1], ylim=[0, 1])
    ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)
    buf.seek(0)
    overlay_img = Image.open(buf)
    original_img = Image.fromarray(image_data_wlut)

    # Resize overlay whilst maintaining aspect ratio
    overlay_width = int(original_img.width * 0.28)
    overlay_height = int(overlay_width * overlay_img.height / overlay_img.width)
    overlay_img = overlay_img.resize((overlay_width, overlay_height))
    overlay_position = (0, original_img.height - overlay_height)
    original_img.paste(overlay_img, overlay_position, overlay_img)
    original_img_cv = np.array(original_img.convert('RGB'))

    # Delete variables
    del overlay_img, original_img, image_data_wlut
    gc.collect()

    # Define labels and font properties
    labels = ["Rec. 2020", "DCI-P3", "Rec. 709"]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    font_thickness = 2

    # Calculate total width and height of the labels
    spacing_width = cv2.getTextSize(" ", font, font_scale, font_thickness)[0][0]
    total_text_width = sum([cv2.getTextSize(label, font, font_scale, font_thickness)[0][0] for label in labels]) + (len(labels) - 1) * spacing_width
    max_text_height = max([cv2.getTextSize(label, font, font_scale, font_thickness)[0][1] for label in labels])

    # Determine the position for text start
    x_text_start = int(original_img_cv.shape[1] * 0.05)
    y_text_start = original_img_cv.shape[0] - max_text_height - int(original_img_cv.shape[0] * 0.01)

    # Calculate width and height of the rectangle
    rectangle_height = int(max_text_height * 1.5)
    rectangle_width = int(total_text_width * 1.03)

    # Compute rectangle start positions such that the text is centered
    rectangle_x_pos = x_text_start - (rectangle_width - total_text_width) // 2
    rectangle_y_pos = y_text_start - (rectangle_height - max_text_height) // 2

    cv2.rectangle(original_img_cv, (rectangle_x_pos, rectangle_y_pos), (rectangle_x_pos + rectangle_width, rectangle_y_pos + rectangle_height), (0, 0, 0), thickness=-1)

    # Draw labels
    x_pos = x_text_start
    y_pos = y_text_start + max_text_height
    for label in labels:
        cv2.putText(original_img_cv, label, (x_pos, y_pos), font, font_scale, (255, 255, 255), thickness=font_thickness, lineType=cv2.LINE_AA)
        text_width, _ = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
        x_pos += text_width + spacing_width

    # Save images
    final_image_path = 'gamut_' + os.path.basename(image_file)
    output_image_path_png = os.path.join(output_directory, os.path.splitext(final_image_path)[0] + '.png')
    cv2.imwrite(output_image_path_png, cv2.cvtColor(original_img_cv, cv2.COLOR_RGB2BGR))

    # Delete variable
    del original_img_cv
    gc.collect()
    num += 1
    print(f"Saved {output_image_path_png} ({round(num / len(image_files) * 100, 1)}% complete)")