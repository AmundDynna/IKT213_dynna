import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, img_bin = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return img_bin

def orb_brief(image_1, image_2):
    img_bin1 = preprocess_image(image_1)
    img_bin2 = preprocess_image(image_2)

    orb = cv2.ORB_create(nfeatures=50000)
    kp1, des1 = orb.detectAndCompute(img_bin1, None)
    kp2, des2 = orb.detectAndCompute(img_bin2, None)
    if des1 is None or des2 is None:
        return 0, None
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.60 * n.distance]

    match_img = cv2.drawMatches(image_1, kp1, image_2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return len(good_matches), match_img


def sift_flann(image_1, image_2):
    img_bin1 = preprocess_image(image_1)
    img_bin2 = preprocess_image(image_2)

    sift = cv2.SIFT_create(nfeatures=10000)
    kp1, des1 = sift.detectAndCompute(img_bin1, None)
    kp2, des2 = sift.detectAndCompute(img_bin2, None)

    if des1 is None or des2 is None:
        return 0, None
    
    index_params = dict(algorithm=1, trees=20)
    search_params = dict(checks=500)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.65 * n.distance]

    match_img = cv2.drawMatches(image_1, kp1, image_2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return len(good_matches), match_img

def data_check():
    img1 = cv2.imread('fingerprint/UiA_front1.png')
    img2 = cv2.imread('fingerprint/UiA_front2.jpg')

    _, match_img = orb_brief(img1, img2)

    plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    _, match_img = sift_flann(img1, img2)
    plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    data_check()