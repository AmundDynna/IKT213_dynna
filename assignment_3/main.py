import cv2
import numpy as np


def sobel_edge_detection(image):
    ksize = (3,3)
    SigmaX = 0
    blur_img = cv2.GaussianBlur(image, ksize, SigmaX)
    
    ksize = 1
    dx = 1
    dy = 1
    sobel_img = cv2.Sobel(blur_img, cv2.CV_64F, dx, dy, ksize)
    cv2.imwrite(f"solution/lambo_sobel.png", sobel_img)
    

def canny_edge_detection(image, threshold_1, threshold_2):
    kisize = (3,3)
    SigmaX = 0
    blur_img = cv2.GaussianBlur(image, kisize, SigmaX)
    
    canny_img = cv2.Canny(blur_img, threshold_1, threshold_2)
    cv2.imwrite(f"solution/lambo_canny.png", canny_img)
    
def template_match(image, template):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    threshold = 0.9
    
    w, h = template_gray.shape[::-1]
    
    res = cv2.matchTemplate(image_gray, template_gray, cv2.TM_CCOEFF_NORMED) # Only method i got realistic results using threshold of 0.9
    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
        
    cv2.imwrite(f"solution/shapes_matched.png", image)
    
def resize(image, scale_factor: int, up_or_down: str):
    amount_scaled = 1
    while amount_scaled < scale_factor:
        if up_or_down == "up":
            image = cv2.pyrUp(image)
        elif up_or_down == "down":
            image = cv2.pyrDown(image)
        amount_scaled *= 2
    cv2.imwrite(f"solution/lambo_scaled.png", image)
    


def main():
    lambo = cv2.imread("lambo.png")
    sobel_edge_detection(lambo)
    canny_edge_detection(lambo, 50, 50)

    shapes = cv2.imread("shapes.png")
    shapes_template = cv2.imread("shapes_template.jpg")
    template_match(shapes, shapes_template)
    resize(lambo, 2, "down")


if __name__ == "__main__":
    main()
