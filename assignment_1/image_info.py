import numpy as np
import cv2

def print_image_information(image):
    # https://stackoverflow.com/questions/19098104/python-opencv2-cv2-wrapper-to-get-image-size
    height, width, channels = image.shape
    
    print("Height:", height)
    print("Width:", width)
    print("Channels:", channels)
    print("Size:", image.size)
    print("Data type:", image.dtype)

img = cv2.imread('assignment_1/lena.png')
print_image_information(img)