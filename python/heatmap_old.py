import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import argparse

def eotf_ST2084(N, m1=0.1593017578125, m2=78.84375, c1=0.8359375, c2=18.8515625, c3=18.6875):
    L = (np.clip(np.power(N, (1/m2)) - c1, 0, None) / (c2 - c3 * np.power(N, (1/m2))))
    return np.nan_to_num(np.power(L, 1/m1), nan=0.0)

def eotf_inverse_ST2084(N, m1=0.1593017578125, m2=78.84375, c1=0.8359375, c2=18.8515625, c3=18.6875):
    Y_p = np.power(N, m1)
    return np.power((c1 + c2 * Y_p) / (c3 * Y_p + 1), m2)

# Create the parser
parser = argparse.ArgumentParser(description='Create an HDR heatmap from an image.')

# Add the arguments
parser.add_argument('InputFolder', metavar='InputFolder', type=str, help='the path to the input folder')
parser.add_argument('OutputFolder', metavar='OutputFolder', type=str, help='the path to the output folder')

# Execute the parse_args() method
args = parser.parse_args()

input_folder = args.InputFolder
output_folder = args.OutputFolder

image_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

if not image_files:
    print("No images found in the input folder. Exiting.")
    raise SystemExit
num = 0
for image in image_files:
    base_name = os.path.basename(image).rsplit('.', 1)[0]
    output_image_path = os.path.join(output_folder, base_name + '-heatmap.png')
    img = plt.imread(image)
    dc = eotf_ST2084(img)
    ll = 0.26270021 * dc[..., 0] + 0.67799807 * dc[..., 1] + 0.05930172 * dc[..., 2]
    llimg = 0.26270021 * img[..., 0] + 0.67799807 * img[..., 1] + 0.05930172 * img[..., 2]

    txt = f"Max light level: {np.max(ll * 10000):.1f} cd/m$^2$\t\tMean light level: {np.mean(ll[ll != 0] * 10000):.1f} cd/m$^2$"

    fig, ax = plt.subplots(figsize=(16, 9))
    cmap = LinearSegmentedColormap.from_list('custom', ['#000000','#000e4d','#001c99','#004cd3','#0087fd','#03e1cf','#d3eb07','#ff450e','#ff0fad','#ffffff'], N=1000)
    im = ax.imshow(llimg, cmap=cmap, vmin=0, vmax=1)

    cbar_labels = [0, 0.1, 1, 5, 10, 48, 100, 250, 500, 750, 1000, 2000, 4000, 10000]
    cbar = fig.colorbar(im, cax=fig.add_axes([0.005, 0.01, 0.015, 0.98]), ticks=eotf_inverse_ST2084(np.array(cbar_labels) / 10000))
    cbar.ax.set_yticklabels(cbar_labels, color='white', fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="none"))

    ax.text(0.5, 0.995, txt, transform=ax.transAxes, fontsize=12, ha='center', va='top', color='white', bbox=dict(facecolor='black', boxstyle='square,pad=0.25'))
    ax.set_axis_off()

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(output_image_path, dpi=240) # change dpi to 240 to output 3840x2160 images
    plt.close(fig)
    num += 1
    print(f"Saved {output_image_path} ({round(num / len(image_files) * 100, 1)}% complete)")
