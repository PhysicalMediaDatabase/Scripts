# Scripts
A collection of scripts used for processing physical media RIPs for analysis

> [!IMPORTANT]
> This software is designed for processing video files ripped from UHD Blurays which are encoded in 2160p Rec. 2100/BT.2100 format (10-bit PQ HDR brightness and BT.2020 color gamut). Do not pass in video files that are not encoded in this format.

### High-level overview
This script suite is designed to do the following:
1. Create unedited, lossless YUV PNGs at 30-second intervals all the way through the video file you supply (the interval can be configured via a third, optional command line argument to `full_workflow.sh`)
2. Create a full-resolution PNG of each screenshot showing an "HDR Heatmap" in which the brightness (Y) values of the image are mapped onto a color spectrum to visualize how well the color grade takes advantage of the HDR brightness range. SDR is defined as 100 nits, so the range of black-to-blue values would be within the SDR brightness range.
3. Create a full-resolution PNG of each screenshot showing the color gamut usage. This is done in two ways:
   - The color values for each pixel are sampled and plotted on a chart within the Rec. 2020 color space, with lines marking the boundaries of the P3 and Rec. 709 color spaces
   - The image itself is tonemapped to Rec. 709 SDR (standard color space) but all pixels that were originally within that standard color space are turned to gray, highlighting the sections of the image that exceeded that color space
4. Create a full-resolution PNG of each screenshot that is simply tonemapped to the Rec. 709 SDR color space for easier viewing
5. Create compressed JPGs of all the processed images (items 2-4)

### How to use
1. Run the `download_lut.sh` script to acquire the necessary HDR-to-SDR lookup table
2. Ensure you have `parallel`, `ffmpeg`, and `python3` installed (the `test_install.sh` script can help with that)
3. Install the necessary python requirements (e.g. `pip install -r requirements.txt`)
4. Run `full_workflow.sh <video> <storagefolder>`. All the necessary organization subfolders will be created within that storage folder, so it is recommended to use the same one every time.
