#!/bin/bash

total_start_time=$(date +%s)

# Check if the correct number of arguments are provided, and if not, print help message
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 video_file output_folder [seconds_per_sample]"
    exit 1
fi



# Default of 30 seconds per sample results in about 240 images for a 2-hour movie, which for 4K video will be at least 7GB of raw PNGs
seconds_per_sample=${3:-30}

echo "Seconds per sample: $seconds_per_sample"



python_script="python/convert.py"

# Check if the Python script exists
if [ ! -f "$python_script" ]; then
    echo "Error: Python script $python_script not found!"
    exit 1
fi



# Get the video file and output folder from the command line arguments
video_file="$1"
output_folder="$2"

# Check if the video file exists
if [ ! -f "$video_file" ]; then
    echo "Error: Video file $video_file not found!"
    exit 1
fi

# Check if the output folder exists
if [ ! -d "$output_folder" ]; then
    echo "Error: Output folder $output_folder not found!"
    exit 1
fi



# Get the base name of the video file (without path and extension)
base_name=$(basename "$video_file" .${video_file##*.})

echo "Base name: $base_name"

# subfolder names
subfolder_raw="raw"
subfolder_heatmaps="heatmaps"
subfolder_gamut="gamut"
subfolder_tonemapped="tonemapped"

# Create subfolder pathnames
subfolder_raw_fullpath="$output_folder/$base_name/$subfolder_raw"
subfolder_heatmaps_fullpath="$output_folder/$base_name/$subfolder_heatmaps"
subfolder_gamut_fullpath="$output_folder/$base_name/$subfolder_gamut"
subfolder_tonemapped_fullpath="$output_folder/$base_name/$subfolder_tonemapped"

# Create the subfolders
mkdir -p $subfolder_raw_fullpath
mkdir -p $subfolder_heatmaps_fullpath
mkdir -p $subfolder_gamut_fullpath
mkdir -p $subfolder_tonemapped_fullpath

echo "Raw PNG directory: $subfolder_raw_fullpath"
echo "Heatmaps directory: $subfolder_heatmaps_fullpath"
echo "Gamut view directory: $subfolder_gamut_fullpath"
echo "Tonemapped directory: $subfolder_tonemapped_fullpath"

is_folder_empty() {
    if [ -z "$(ls -A "$1")" ]; then
        return 0  # Folder is empty
    else
        return 1  # Folder is not empty
    fi
}

ffmpeg_message=""

# Extract frames from the video every specified seconds, time the command, and save the message
if is_folder_empty "$subfolder_raw_fullpath"; then
    ffmpeg_start_time=$(date +%s)
    ffmpeg -i "$video_file" -vf "fps=1/$seconds_per_sample" "$subfolder_raw_fullpath/%03d.png"
    ffmpeg_end_time=$(date +%s)
    ffmpeg_elapsed_time=$((end_time - start_time))
    ffmpeg_hours=$((elapsed_time / 3600))
    ffmpeg_minutes=$(((elapsed_time % 3600) / 60 ))
    ffmpeg_seconds=$((elapsed_time % 60))
    ffmpeg_message=$(printf "ffmpeg command took %02d:%02d:%02d (hh:mm:ss)\n" $ffmpeg_hours $ffmpeg_minutes $ffmpeg_seconds)
else
    ffmpeg_message="Raw PNG directory is not empty. Skipping ffmpeg command."
fi
echo "$ffmpeg_message"

# Run the Python scripts
python3 $python_script hdr $subfolder_raw_fullpath $subfolder_heatmaps_fullpath
python3 $python_script wcg $subfolder_raw_fullpath $subfolder_gamut_fullpath
python3 $python_script map $subfolder_raw_fullpath $subfolder_tonemapped_fullpath

# Compress files
mkdir -p $subfolder_heatmaps_fullpath-compressed
mkdir -p $subfolder_gamut_fullpath-compressed
mkdir -p $subfolder_tonemapped_fullpath-compressed
find $subfolder_heatmaps_fullpath -name '*.png' | parallel convert "{}" -quality 85 "$subfolder_heatmaps_fullpath-compressed/{/.}.jpg"
find $subfolder_gamut_fullpath -name '*.png' | parallel convert "{}" -quality 85 "$subfolder_gamut_fullpath-compressed/{/.}.jpg"
find $subfolder_tonemapped_fullpath -name '*.png' | parallel convert "{}" -quality 85 "$subfolder_tonemapped_fullpath-compressed/{/.}.jpg"

# Print the timing of the entire script
total_end_time=$(date +%s)
total_elapsed_time=$((total_end_time - total_start_time))
total_hours=$((total_elapsed_time / 3600))
total_minutes=$(((total_elapsed_time % 3600) / 60 ))
total_seconds=$((total_elapsed_time % 60))
echo "ffmpeg message was: $ffmpeg_message"
printf "Total time elapsed: %02d:%02d:%02d (hh:mm:ss)\n" $total_hours $total_minutes $total_seconds