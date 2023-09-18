import cv2
import numpy as np
import matplotlib.pyplot as plt

template_image1_path = 'template_notebook.jpg'
reference_image1_path = 'find_notebook_pose.jpg'
template_image2_path = 'template_tea_cup_pad.jpg'
reference_image2_path = 'find_tea_cup_pad_pose.jpg'


def remove_white_background(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_not(mask)
    result = cv2.bitwise_and(image, image, mask=mask)
    return result


def find_the_area_in_space(template_path, reference_path):
    template_image = cv2.imread(template_path)
    reference_image = cv2.imread(reference_path)

    if template_image is None or reference_image is None:
        print("Error: One or both images could not be loaded.")
    else:
        template_image = remove_white_background(template_image)
        reference_image = remove_white_background(reference_image)

        template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
        reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create()

        key_points_template, descriptors_template = sift.detectAndCompute(template_gray, None)
        key_points_reference, descriptors_reference = sift.detectAndCompute(reference_gray, None)

        bf = cv2.BFMatcher()

        matches = bf.knnMatch(descriptors_template, descriptors_reference, k=2)

        good_matches = []
        ratio_threshold = 0.75
        for m, n in matches:
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)

        src_pts = np.float32([key_points_template[m.queryIdx].pt for m in good_matches])
        dst_pts = np.float32([key_points_reference[m.trainIdx].pt for m in good_matches])

        homography_matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        transformation_matrix = np.zeros((4, 4), dtype=np.float32)
        transformation_matrix[:3, :3] = homography_matrix
        transformation_matrix[3, 3] = 1

        model_pts = np.array([[0, 0, 0], [0, 210, 0], [297, 210, 0], [297, 0, 0]], dtype=np.float32)

        ones_column = np.ones((model_pts.shape[0], 1), dtype=np.float32)
        model_pts = np.hstack((model_pts, ones_column))

        projected_pts = np.dot(transformation_matrix, model_pts.T).T

        projected_pts = projected_pts[:, :2] / projected_pts[:, 3:]

        rotation_matrix = transformation_matrix[:3, :3] / np.linalg.norm(transformation_matrix[:3, :3], axis=0)
        translation_vector = transformation_matrix[:3, 3]

        print("Translation vector (X, Y, Z):", translation_vector)
        print("Rotation matrix:\n", rotation_matrix)

        notebook_length = 297
        notebook_width = 210
        notebook_height = 10

        cuboid_vertices = np.array([
            [0, 0, 0],
            [notebook_length, 0, 0],
            [notebook_length, notebook_width, 0],
            [0, notebook_width, 0],
            [0, 0, -notebook_height],
            [notebook_length, 0, -notebook_height],
            [notebook_length, notebook_width, -notebook_height],
            [0, notebook_width, -notebook_height]
        ], dtype=np.float32)

        transformed_cuboid = np.dot(rotation_matrix, cuboid_vertices.T).T + translation_vector.T

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        cuboid_edges = [
            [transformed_cuboid[0], transformed_cuboid[1], transformed_cuboid[2], transformed_cuboid[3],
             transformed_cuboid[0]],
            [transformed_cuboid[4], transformed_cuboid[5], transformed_cuboid[6], transformed_cuboid[7],
             transformed_cuboid[4]],
            [transformed_cuboid[0], transformed_cuboid[4]],
            [transformed_cuboid[1], transformed_cuboid[5]],
            [transformed_cuboid[2], transformed_cuboid[6]],
            [transformed_cuboid[3], transformed_cuboid[7]]
        ]

        for edge in cuboid_edges:
            edge = np.array(edge)
            ax.plot(edge[:, 0], edge[:, 1], edge[:, 2])

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()


find_the_area_in_space(template_image1_path, reference_image1_path)
find_the_area_in_space(template_image2_path, reference_image2_path)
