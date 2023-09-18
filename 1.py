import cv2
import matplotlib.pyplot as plt

template_image1_path = cv2.imread('template_notebook.jpg', cv2.IMREAD_GRAYSCALE)
reference_image1_path = cv2.imread('find_notebook_pose.jpg', cv2.IMREAD_GRAYSCALE)
template_image2_path = cv2.imread('template_tea_cup_pad.jpg', cv2.IMREAD_GRAYSCALE)
reference_image2_path = cv2.imread('find_tea_cup_pad_pose.jpg', cv2.IMREAD_GRAYSCALE)


def find_key_points(template_path, reference_path):
    if template_path is None or reference_path is None:
        print("Error: One or both images could not be loaded.")
    else:
        sift = cv2.SIFT_create()

        key_points_template, descriptors_template = sift.detectAndCompute(template_path, None)
        key_points_reference, descriptors_reference = sift.detectAndCompute(reference_path, None)

        bf = cv2.BFMatcher()

        matches = bf.knnMatch(descriptors_template, descriptors_reference, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        matched_image = cv2.drawMatches(template_path, key_points_template, reference_path, key_points_reference,
                                        good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        plt.figure(figsize=(10, 5))
        plt.imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
        plt.title('Matched Key points')
        plt.show()


find_key_points(template_image1_path, reference_image1_path)
find_key_points(template_image2_path, reference_image2_path)
