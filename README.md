# ImageMosaicBuilder
This script performs feature-based image matching, geometric relationship estimation, and image mosaic generation between pairs of images in a given directory. It uses OpenCV (cv2) to detect features, compute fundamental and homography matrices, and blend overlapping images into a seamless mosaic.

The program takes as input:

A directory of images.

An output directory for saving results.

It then:

Reads and sorts all .jpg images from the input folder.

Extracts keypoints and descriptors from each image using the SIFT (Scale-Invariant Feature Transform) algorithm.

Matches keypoints between all possible pairs of images using a Brute-Force Matcher with ratio testing.

Computes the Fundamental Matrix (F) between matched pairs to verify if the images belong to the same scene and to find epipolar constraints.

Draws epipolar lines to visualize geometric relationships between the two images.

Estimates a Homography Matrix (H) if enough inliers exist, determining whether the two images can be aligned (e.g., are parts of the same panorama).

Generates an image mosaic by warping one image onto another using the homography and blending overlapping regions smoothly.

Saves all outputs (matched images, epipolar lines, homography visualizations, and mosaics) into the specified output folder.

Logs all processing details and results to an output.txt file in the output directory.

Each image pair produces several outputs in the output_directory:

matched_<image1>_<image2>.jpg – Visualization of SIFT matches.

fundamental_matrix_<image1>_<image2>.jpg – Inlier matches after Fundamental Matrix computation.

epipolar_lines_<image1>_<image2>.jpg – Image with drawn epipolar lines.

homography_<image1>_<image2>.jpg – Inlier matches after Homography estimation.

mosaic_<image1>_<image2>.jpg – Final blended mosaic (if alignment succeeds).

output.txt – Detailed processing log with match statistics and decisions.


python script.py <image_directory> <output_directory>


