import cv2
import numpy as np
import matplotlib.pyplot as plt

template_image1_path = cv2.imread('template_notebook.jpg')
reference_image1_path = cv2.imread('find_notebook_pose.jpg')
template_image2_path = cv2.imread('template_tea_cup_pad.jpg')
reference_image2_path = cv2.imread('find_tea_cup_pad_pose.jpg')


def remove_white_background(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_not(mask)
    result = cv2.bitwise_and(image, image, mask=mask)
    return result


def find_image(template_path, reference_path):
    if template_path is None or reference_path is None:
        print("Error: One or both images could not be loaded.")
    else:
        template_path = remove_white_background(template_path)
        reference_path = remove_white_background(reference_path)

        sift = cv2.SIFT_create()

        key_points_template, descriptors_template = sift.detectAndCompute(template_path, None)
        key_points_reference, descriptors_reference = sift.detectAndCompute(reference_path, None)

        bf = cv2.BFMatcher()

        matches = bf.knnMatch(descriptors_template, descriptors_reference, k=2)

        good_matches = []
        ratio_threshold = 0.75
        for m, n in matches:
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)

        src_pts = np.float32([key_points_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([key_points_reference[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        homography_matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        aligned_template = cv2.warpPerspective(template_path, homography_matrix,
                                               (reference_path.shape[1], reference_path.shape[0]))

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(template_path, cmap='gray')
        plt.title('Template Image')

        plt.subplot(1, 2, 2)
        plt.imshow(aligned_template, cmap='gray')
        plt.title('Aligned Template Image')
        plt.show()


find_image(template_image1_path, reference_image1_path)
find_image(template_image2_path, reference_image2_path)
