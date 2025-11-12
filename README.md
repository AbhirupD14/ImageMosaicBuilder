# ImageMosaicBuilder

This project performs **feature-based image matching**, **geometric relationship estimation**, and **image mosaic generation** between pairs of images in a given directory.  
It uses **OpenCV (cv2)** to detect image features, compute **Fundamental** and **Homography** matrices, and blend overlapping images into a seamless mosaic.

---

## Input

The program takes as input:
- A directory of images
- An output directory for saving results

---

## Process Overview

The script performs the following steps:

1. **Read and sort** all `.jpg` images from the input folder  
2. **Extract keypoints and descriptors** using the **SIFT** (Scale-Invariant Feature Transform) algorithm  
3. **Match keypoints** between all possible pairs of images using a **Brute-Force Matcher** with ratio testing  
4. **Compute the Fundamental Matrix (F)** between matched pairs to determine if the images depict the same scene  
5. **Draw epipolar lines** to visualize geometric relationships between image pairs  
6. **Estimate a Homography Matrix (H)** if enough inliers exist, determining whether the two images can be aligned (e.g., parts of the same panorama)  
7. **Generate an image mosaic** by warping one image onto another using the homography and blending overlapping regions smoothly  
8. **Save all results** (matched images, epipolar lines, homography visualizations, and mosaics) to the output directory  
9. **Log processing details** and results to an `output.txt` file  

---

## Output Files

Each image pair produces several outputs in the specified `output_directory`:

| Output File | Description |
|--------------|-------------|
| `matched_<image1>_<image2>.jpg` | Visualization of SIFT feature matches |
| `fundamental_matrix_<image1>_<image2>.jpg` | Inlier matches after Fundamental Matrix computation |
| `epipolar_lines_<image1>_<image2>.jpg` | Image with drawn epipolar lines |
| `homography_<image1>_<image2>.jpg` | Inlier matches after Homography estimation |
| `mosaic_<image1>_<image2>.jpg` | Final blended mosaic (if alignment succeeds) |
| `output.txt` | Detailed processing log with match statistics and alignment decisions |

---

## Usage

```bash
python script.py <image_directory> <output_directory>
