import numpy as np
import matplotlib.pyplot as plt
from common import eotf_ST2084, eotf_inverse_ST2084

# Define empty function signature
# TODO - copy from heatmap.py
def hdr_heatmap(input_image, output_image):
    
    # Load image data
    image_data = plt.imread(input_image)

    # Apply the EOTF to the image data
    # TODO - Why are we not scaling this image data before applying EOTF as compared to the gamut convert?
    decoded_color = eotf_ST2084(image_data)

    # Calculate the linear light levels of the decoded image data
    # TODO - What exactly does this conversion do?
    linear_light = 0.26270021 * decoded_color[..., 0] + 0.67799807 * decoded_color[..., 1] + 0.05930172 * decoded_color[..., 2]

    # Calculate the linear light levels of the original image data
    linear_light_image = 0.26270021 * image_data[..., 0] + 0.67799807 * image_data[..., 1] + 0.05930172 * image_data[..., 2]

    # Create a string that describes the maximum and mean light levels in the image
    text_description = f"Max light level: {np.max(linear_light * 10000):.1f} cd/m$^2$\t\tMean light level: {np.mean(linear_light[linear_light != 0] * 10000):.1f} cd/m$^2$"

    # Create the heatmap image ------------------------------------------------

    # Create a new figure and subplot with specified size
    fig, ax = plt.subplots(figsize=(16, 9))

    # Create a custom colormap for the heatmap
    custom_cmap = plt.cm.colors.LinearSegmentedColormap.from_list('custom', ['#000000','#000e4d','#001c99','#004cd3','#0087fd','#03e1cf','#d3eb07','#ff450e','#ff0fad','#ffffff'], N=1000)

    # Display the data as an image in the subplot using the custom colormap
    image = ax.imshow(linear_light_image, cmap=custom_cmap, vmin=0, vmax=1)

    # Add colorbar to the image -----------------------------------------------

    # Define the colorbar labels
    colorbar_labels = [0, 0.1, 1, 5, 10, 48, 100, 250, 500, 750, 1000, 2000, 4000, 10000]

    # Create a colorbar with the specified labels
    colorbar = fig.colorbar(image, cax=fig.add_axes([0.005, 0.01, 0.015, 0.98]), ticks=eotf_inverse_ST2084(np.array(colorbar_labels) / 10000))

    # Set the colorbar tick labels to the specified values
    colorbar.ax.set_yticklabels(colorbar_labels, color='white', fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="none"))

    # Add text description to the image ---------------------------------------

    # Add the text description to the image
    ax.text(0.5, 0.995, text_description, transform=ax.transAxes, fontsize=12, ha='center', va='top', color='white', bbox=dict(facecolor='black', boxstyle='square,pad=0.25'))

    # Remove the axis from the image
    ax.set_axis_off()

    # Save the image ----------------------------------------------------------

    # Adjust the layout of the figure
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Save the current figure as an image
    plt.savefig(output_image, dpi=240)

    # Close the figure
    plt.close(fig)

    return