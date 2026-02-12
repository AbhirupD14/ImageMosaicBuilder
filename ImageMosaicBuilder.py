import cv2
import os
import sys
import numpy as np
import random
import shutil

# Util function taken from previous homework
def get_images(img_dir):
    start_cwd = os.getcwd()
    os.chdir(img_dir)
    img_name_list = os.listdir('./')
    img_name_list = [name for name in img_name_list if 'jpg' in name.lower()]
    img_name_list.sort()

    img_list = []
    img_color_list = []
    for i_name in img_name_list:
        im = cv2.imread(i_name, cv2.IMREAD_GRAYSCALE)
        im_color = cv2.imread(i_name, cv2.IMREAD_COLOR)
        if im is None:
            print('Could not open', i_name)
            sys.exit(0)
        img_list.append(im)
        img_color_list.append(im_color)

    os.chdir(start_cwd)
    return img_name_list, img_list, img_color_list

def draw_epipolar_lines(img, pts, F, output_dir, img_name):
    if len(img.shape) == 2:  # Grayscale image
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Converting to draw epipolar lines in color

    img_lines = img.copy()

    for _, pt in enumerate(pts):
        # Compute the epipolar line for the point
        # Line equation: F * point
        line = np.dot(F, np.append(pt, 1))

        # Calculate two points on the line
        x0, y0 = map(int, [0, -line[2] / line[1]])  # Point at x = 0
        x1, y1 = map(int, [img.shape[1], -(line[2] + line[0] * img.shape[1]) / line[1]])  # Point at x = img_width

        # Generate a random color for each line
        color = tuple([random.randint(0, 255) for _ in range(3)])  # Random color in BGR format

        # Draw the epipolar line on the image
        cv2.line(img_lines, (x0, y0), (x1, y1), color, 1)  # Draw with the random color

    # Save the image with epipolar lines
    output_path = os.path.join(output_dir, f"epipolar_lines_{img_name}")
    cv2.imwrite(output_path, img_lines)

    return img_lines

def extract_and_match(image1, image2, name1, name2, output_dir):
    sift = cv2.SIFT_create()
    k1, d1 = sift.detectAndCompute(image1, None)
    k2, d2 = sift.detectAndCompute(image2, None)

    print(f"Number of keypoints for {name1}: {len(k1)}")
    print(f"Number of keypoints for {name2}: {len(k2)}")

    # Create BFMatcher (Brute-Force Matcher) with default parameters
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # Match descriptors
    matches = bf.match(d1, d2)

    bf = cv2.BFMatcher(cv2.NORM_L2)

    # Find the two best matches for each descriptor
    matches = bf.knnMatch(d1, d2, k=2)

    # Keep only matches where the best match is significantly better than the second-best match (ratio < 0.8)
    best_matches = [m for m, n in matches if m.distance < 0.8 * n.distance]

    matched_img = cv2.drawMatches(image1, k1, image2, k2, best_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    match_rate1 = (len(best_matches) / len(k1))
    match_rate2 = (len(best_matches) / len(k2))

    match_criteria = min(match_rate1, match_rate2)
    
    print(f"Number of matches: {len(best_matches)}")
    print(f"Fraction of keypoints with matches in {name1}: {match_rate1:.2f}")
    print(f"Fraction of keypoints with matches in {name2}: {match_rate2:.2f}")

    # Save the matched image
    output_path = os.path.join(output_dir, f"matched_{name1}_{name2}.jpg")
    cv2.imwrite(output_path, matched_img)

    if match_criteria < 0.05 and len(best_matches) < 200:
        print(f"Percentage of keypoint matches too low for {name1}, {name2}: {match_criteria:.2f}")
        print(f"Images {name1}, {name2} are not of same scene, stopping consideration\n")
        return
    
    return [(image1, k1, name1), (image2, k2, name2), best_matches]

def fundMat(img1_info, img2_info, good_matches, output_dir):
    pts1 = np.float32([img1_info[1][m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([img2_info[1][m.trainIdx].pt for m in good_matches])

    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 5.0)

    inlier_pts1 = pts1[mask.ravel() == 1]
    inlier_pts2 = pts2[mask.ravel() == 1]

    num_inliers = np.sum(mask)
    total_matches = len(good_matches)
    inlier_percentage = (num_inliers / total_matches)

    print(f"Fundamental matrix computed. Inliers: {num_inliers} out of {total_matches} matches.\nFundamental Matrix Matched inlier percentage: {inlier_percentage:.2f}")

    img_matches = cv2.drawMatches(img1_info[0], img1_info[1], img2_info[0], img2_info[1], good_matches, None, matchColor=(0, 255, 0), # Green for inliers
                                  singlePointColor=(255, 0, 0), # Red for keypoints
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Save the fundamental matrix matched image
    output_path = os.path.join(output_dir, f"fundamental_matrix_{img1_info[2]}_{img2_info[2]}.jpg")
    cv2.imwrite(output_path, img_matches)

    if inlier_percentage < .5:
        print(f"Inlier percentage for {img1_info[2]}, {img2_info[2]} is too low: {inlier_percentage:.2f}\nDropping image pair from consideration\n")
        return None
    
    # Draw epipolar lines for inliers
    print("Displaying Epipolar Lines")
    img_epilines = draw_epipolar_lines(img2_info[0], inlier_pts1, F, output_dir, f"epipolar_lines_{img1_info[2]}_{img2_info[2]}.jpg")

    return [img1_info, img2_info, good_matches, num_inliers]

def estimate_homography(img1_info, img2_info, good_matches, output_dir):
    # Extract keypoint coordinates for the inlier matches
    pts1 = np.float32([img1_info[1][m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([img2_info[1][m.trainIdx].pt for m in good_matches])

    # Compute the homography matrix using RANSAC
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

    if H is None:
        print("Homography estimation failed.")
        return

    # Count inliers
    inliers = mask.ravel().tolist()
    num_inliers = np.sum(mask)
    total_matches = len(good_matches)
    inlier_percentage = (num_inliers / total_matches)

    print(f"Homography matrix computed.\nHomography Matrix Inliers: {num_inliers} out of {total_matches} matches.")
    print(f"Homography Matrix Inlier match percentage: {inlier_percentage:.2f}\n")

    # Draw only the inlier matches with random colors
    inlier_matches = [good_matches[i] for i in range(len(good_matches)) if inliers[i]]

    # Create an output image showing the inlier matches with different colors
    matched_img = np.zeros_like(img1_info[0])  # Create an empty canvas for visualization
    matched_img = cv2.drawMatches(
        img1_info[0], img1_info[1], img2_info[0], img2_info[1], inlier_matches, None,
        matchColor=None,  # Let OpenCV choose colors
        singlePointColor=(0, 255, 0),  # Green for keypoints
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # Save the homography matched image
    output_path = os.path.join(output_dir, f"homography_{img1_info[2]}_{img2_info[2]}.jpg")
    cv2.imwrite(output_path, matched_img)

    return H, num_inliers

def decide_alignment(fund_inliers, homography_inliers):
    match_ratio = homography_inliers / fund_inliers if fund_inliers > 0 else 0

    if match_ratio > 0.40:  
        return "Yes", f"Percent of fundamental inliers retained in the homography estimation: {match_ratio:.2f}.\nThe images can be accurately aligned."
    else:
        return "No", f"Percent of fundamental inliers retained in the homography estimation: {match_ratio:.2f}.\nThe images cannot be accurately aligned."

def create_mosaic(img1, img2, H, output_dir, img1_name, img2_name):
    # Convert images to BGR if grayscale
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR) if len(img1.shape) == 2 else img1
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR) if len(img2.shape) == 2 else img2

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Transform img1's corners to find output dimensions
    corners_img1 = np.array([[0, 0], [w1, 0], [w1, h1], [0, h1]], dtype=np.float32).reshape(-1, 1, 2)
    transformed_corners = cv2.perspectiveTransform(corners_img1, H)

    # Determine bounding box including both images
    min_x = min(0.0, transformed_corners[:, 0, 0].min())
    min_y = min(0.0, transformed_corners[:, 0, 1].min())
    max_x = max(w2, transformed_corners[:, 0, 0].max())
    max_y = max(h2, transformed_corners[:, 0, 1].max())

    # Calculate output dimensions with ceil to avoid cutting off
    output_width = int(np.ceil(max_x - min_x))
    output_height = int(np.ceil(max_y - min_y))

    # Translation matrix to shift the transformed img1 into positive coordinates
    translation_matrix = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])

    # Warp img1 with the combined homography and translation
    warped_img1 = cv2.warpPerspective(img1, translation_matrix @ H, (output_width, output_height))

    # Create mosaic canvas
    mosaic = np.zeros((output_height, output_width, 3), dtype=np.uint8)

    # Calculate the position to place img2 in the translated coordinates
    y1 = int(-min_y)
    y2 = y1 + h2
    x1 = int(-min_x)
    x2 = x1 + w2

    # Place img2 in the mosaic, handling boundary cases
    if x1 < output_width and y1 < output_height and x2 > 0 and y2 > 0:
        # Calculate the valid region for img2
        img2_x_start = max(0, -x1)
        img2_y_start = max(0, -y1)
        img2_x_end = min(w2, output_width - x1)
        img2_y_end = min(h2, output_height - y1)

        mosaic_y_start = max(0, y1)
        mosaic_y_end = mosaic_y_start + (img2_y_end - img2_y_start)
        mosaic_x_start = max(0, x1)
        mosaic_x_end = mosaic_x_start + (img2_x_end - img2_x_start)

        mosaic[mosaic_y_start:mosaic_y_end, mosaic_x_start:mosaic_x_end] = img2[img2_y_start:img2_y_end, img2_x_start:img2_x_end]

    # Create masks for blending
    mask1 = (cv2.cvtColor(warped_img1, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)
    mask2 = (cv2.cvtColor(mosaic, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)

    # Overlap mask (both images have content)
    overlap_mask = cv2.bitwise_and(mask1, mask2)
    # Mask for areas only in warped_img1
    mask1_only = cv2.bitwise_and(mask1, cv2.bitwise_not(mask2))

    # Create alpha mask for blending
    alpha = np.zeros_like(mask1, dtype=np.float32)

    # Apply Gaussian blur to overlap region for smooth transition
    if overlap_mask.sum() > 0:
        blurred_overlap = cv2.GaussianBlur(overlap_mask.astype(np.float32), (21, 21), 10)
        blurred_overlap = (blurred_overlap / blurred_overlap.max()) if blurred_overlap.max() > 0 else blurred_overlap
        alpha[overlap_mask > 0] = blurred_overlap[overlap_mask > 0]

    # Set alpha to 1 for areas only in warped_img1
    alpha[mask1_only > 0] = 1.0

    # Expand alpha to 3 channels
    alpha = cv2.merge([alpha, alpha, alpha])

    # Blend images
    mosaic = (warped_img1 * alpha + mosaic * (1 - alpha)).astype(np.uint8)

    # Save the mosaic image
    output_path = os.path.join(output_dir, f"mosaic_{img1_name}_{img2_name}.jpg")
    cv2.imwrite(output_path, mosaic)

    return mosaic

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <image_directory> <output_directory>")
        sys.exit(1)

    image_dir = sys.argv[1]
    output_dir = sys.argv[2]

    # Delete an existing one
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # Create the output directory
    os.makedirs(output_dir)

    # Redirect all print statements to output.txt
    output_file_path = os.path.join(output_dir, "output.txt")
    with open(output_file_path, "w") as f:
        sys.stdout = f  # Redirect stdout to the file

        img_names, img_list, img_color_list = get_images(image_dir)
        
        counter = 0
        
        for i in range(len(img_list)):
            for j in range(i, len(img_list)):
                if i == j:
                    continue
                print(f"Now considering {img_names[i]}, {img_names[j]}")
                print("-"*50)
                mat_info = extract_and_match(img_list[i], img_list[j], img_names[i], img_names[j], output_dir)

                if mat_info == None:
                    continue
                fundMatInfo = fundMat(mat_info[0], mat_info[1], mat_info[2], output_dir)

                if fundMatInfo == None:
                    continue

                H, homography_inliers = estimate_homography(fundMatInfo[0], fundMatInfo[1], fundMatInfo[2], output_dir)
                decision, reason = decide_alignment(fundMatInfo[3], homography_inliers)
                print(decision + "," + reason)

                if decision == "Yes":
                    create_mosaic(img_color_list[i], img_color_list[j], H, output_dir, img_names[i], img_names[j])

        sys.stdout = sys.__stdout__  # Restore stdout to its original value