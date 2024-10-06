import argparse
import os
import time
from gamut import wcg_visualize
from heatmap import hdr_heatmap
from tonemap import tonemap_image

# Define the function mapping
function_mapping = {
    'wcg': [wcg_visualize, 'gamut'],
    'hdr': [hdr_heatmap, 'heatmap'],
    'map': [tonemap_image, 'tonemap']
}

# Define the command line argument parser
parser = argparse.ArgumentParser(description='Run a function on an input and output folder.')
parser.add_argument('function', help='The name of the function to run.')
parser.add_argument('input_folder', help='The path to the input folder.')
parser.add_argument('output_folder', help='The path to the output folder.')

# Parse the command line arguments
args = parser.parse_args()
function_name = args.function
input_folder = args.input_folder
output_folder = args.output_folder

# Get the function from the mapping
function = function_mapping.get(function_name)[0]

# Find all image files in input directory
input_image_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.png')]

# Throw error if no image files in input directory
if not input_image_paths:
    print("No images found in the input folder. Exiting.")
    raise SystemExit

# Start a counter so we can show progress
image_process_count = 0
image_operation_count = 0
total_images = len(input_image_paths)

# Record the start time
start_time = time.time()

# Loop through each image file
for input_image_path in input_image_paths:

    # Increment the counter and print progress
    image_process_count += 1

    # Record the start time of this iteration
    iteration_start_time = time.time()

    # Grab the basename of the image file
    base_name = os.path.basename(input_image_path).rsplit('.', 1)[0]

    # Grab the file suffix for this function
    function_suffix = function_mapping.get(function_name)[1]

    # Define the output image path
    output_image_path = os.path.join(output_folder, base_name + f'-{function_suffix}.png')

    # Check if the output image path already exists
    if os.path.exists(output_image_path):
        print(f"[convert] [{function_name}] ({image_process_count:03d}/{total_images}) Skipped {output_image_path} (already exists)")
        continue

    # Run the function on the image and save the output
    function(input_image_path, output_image_path)

    # Increment the operation counter
    image_operation_count += 1

    # Calculate the time taken for this iteration
    iteration_time = time.time() - iteration_start_time

    # Estimate the remaining time
    remaining_time = iteration_time * (total_images - image_process_count)

    # Convert remaining time to hours, minutes, and seconds
    remaining_hours = int(remaining_time // 3600)
    remaining_minutes = int((remaining_time % 3600) // 60)
    remaining_seconds = int(remaining_time % 60)

    # Print progress and estimated time remaining
    print(f"[convert] [{function_name}] ({image_process_count:03d}/{total_images}) Saved {output_image_path} ({round(image_process_count / total_images * 100, 1):5.1f}% complete, "
          f"estimated time remaining: {remaining_hours}h {remaining_minutes}m {remaining_seconds}s)")

# Print total time taken
total_time = time.time() - start_time
total_hours = int(total_time // 3600)
total_minutes = int((total_time % 3600) // 60)
total_seconds = int(total_time % 60)
print(f"[convert] [{function_name}] {image_operation_count} operations completed, total time taken: {total_hours}h {total_minutes}m {total_seconds}s")