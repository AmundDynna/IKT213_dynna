import cv2
import numpy as np



def find_corners(reference_image):
    gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)
    dst = cv2.dilate(dst,None)
    reference_image[dst>0.01*dst.max()]=[0,0,255]
    cv2.imwrite(f"solution/harris_corners.png", reference_image)


def SIFT(image, align_image):
    max_features = 10
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    align_image_gray = cv2.cvtColor(align_image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(image_gray,None)
    kp2, des2 = sift.detectAndCompute(align_image_gray,None)
    
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    good_sorted = sorted(good, key=lambda x: x.distance)[:max_features]
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   flags = 2,
                   matchesThickness=5)
    
    img_matches = cv2.drawMatches(image,kp1,align_image,kp2,good_sorted,None,**draw_params)
    
    matched_kp1 = [kp1[m.queryIdx] for m in good_sorted]
    reference_matches = cv2.drawKeypoints(image, matched_kp1, None, color=(0,0,255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    matched_kp2 = [kp2[m.trainIdx] for m in good_sorted]
    align_image_matches = cv2.drawKeypoints(align_image, matched_kp2, None, color=(0,0,255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(f"solution/sift_matches.png", img_matches)
    cv2.imwrite(f"solution/sift_reference.png", reference_matches)
    cv2.imwrite(f"solution/sift_align.png", align_image_matches)


if __name__ == "__main__":
    reference_img = cv2.imread("reference_img.png")
    align_img = cv2.imread("align_this.jpg")
    find_corners(reference_img)
    reference_img = cv2.imread("reference_img.png")
    SIFT(reference_img, align_img)