import cv2
import numpy as np
import matplotlib.pyplot as plt

template_image1_path = 'template_notebook.jpg'
reference_image1_path = 'find_notebook_pose.jpg'
template_image2_path = 'template_tea_cup_pad_2.jpg'
reference_image2_path = 'find_tea_cup_pad_pose.jpg'
template_image3_path = 'template_tea_cup_pad.jpg'
reference_image3_path = 'find_tea_cup_pad_pose.jpg'


def find_on_picture(template_path, reference_path):
    template_image = cv2.imread(template_path)
    reference_image = cv2.imread(reference_path)

    if template_image is None or reference_image is None:
        print("Error: One or both images could not be loaded.")
    else:
        template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
        reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()

        key_points_template, descriptors_template = sift.detectAndCompute(template_gray, None)
        key_points_reference, descriptors_reference = sift.detectAndCompute(reference_gray, None)

        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(descriptors_template, descriptors_reference, k=2)

        good_matches = []
        ratio_threshold = 0.75
        for m, n in matches:
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)

        src_pts = np.float32([key_points_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([key_points_reference[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        homography_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        template_corners = np.array(
            [[0, 0], [template_image.shape[1], 0], [template_image.shape[1], template_image.shape[0]],
             [0, template_image.shape[0]]],
            dtype=np.float32).reshape(-1, 1, 2)

        projected_corners = cv2.perspectiveTransform(template_corners, homography_matrix)

        reference_with_cuboid = reference_image.copy()
        colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]
        for i in range(4):
            pt1 = tuple(map(int, projected_corners[i][0]))
            pt2 = tuple(map(int, projected_corners[(i + 1) % 4][0]))
            cv2.line(reference_with_cuboid, pt1, pt2, colors[i % 3], 7)

        cuboid_height = 100

        for i in range(4):
            pt1 = tuple(map(int, projected_corners[i][0]))
            pt2 = (pt1[0], pt1[1] - cuboid_height)
            cv2.line(reference_with_cuboid, pt1, pt2, colors[i % 3], 7)

        for i in range(4):
            pt1 = tuple(map(int, projected_corners[i][0]))
            pt2 = tuple(map(int, projected_corners[i - 4][0]))
            cv2.line(reference_with_cuboid, pt1, pt2, (255, 255, 0), 7)

        upper_corners = []
        for i in range(4):
            pt1 = tuple(map(int, projected_corners[i][0]))
            pt2 = (pt1[0], pt1[1] - cuboid_height)
            upper_corners.append(pt2)

        for i in range(4):
            pt1 = upper_corners[i]
            pt2 = upper_corners[(i + 1) % 4]
            cv2.line(reference_with_cuboid, pt1, pt2, colors[i % 3], 7)

        for corner in upper_corners:
            cv2.circle(reference_with_cuboid, tuple(map(int, corner)), 5, (255, 255, 0), -1)

        plt.imshow(cv2.cvtColor(reference_with_cuboid, cv2.COLOR_BGR2RGB))
        plt.show()


find_on_picture(template_image1_path, reference_image1_path)
find_on_picture(template_image2_path, reference_image2_path)
find_on_picture(template_image3_path, reference_image3_path)
