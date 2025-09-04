import numpy as np
import cv2


def padding(image, border_width):
    padded_image = cv2.copyMakeBorder(image, border_width, border_width, border_width, border_width, cv2.BORDER_REFLECT)
    cv2.imwrite(f"{default_path}/lena_padded.png", padded_image)

def crop(img, x_0, x_1, y_0, y_1):
    cropped_image = img[x_0:x_1, y_0:y_1]
    cv2.imwrite(f"{default_path}/lena_cropped.png", cropped_image)

def resize(image, width, height):
    resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(f"{default_path}/lena_resized.png", resized_image)

def copy(image, empty_picture_array):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            empty_picture_array[i, j] = image[i, j]
    cv2.imwrite(f"{default_path}/lena_copied.png", empty_picture_array)

def grayscale(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f"{default_path}/lena_gray.png", img_gray)

def hsv(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imwrite(f"{default_path}/lena_hsv.png", hsv_image)

def hue_shifted(image, empty_picture_array, hue):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixel = image[i, j]
            b, g, r = int(pixel[0]), int(pixel[1]), int(pixel[2])
            if hue > 0:
                blue = min(b + hue, 255)
                green = min(g + hue, 255)
                red = min(r + hue, 255)
            else:
                blue = max(b + hue, 0)
                green = max(g + hue, 0)
                red = max(r + hue, 0)
            empty_picture_array[i, j] = [blue, green, red]
    cv2.imwrite(f"{default_path}/lena_hue_shifted.png", empty_picture_array)

def smoothing(image):
    kernel_size = 15
    smoothed_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), cv2.BORDER_DEFAULT) 
    cv2.imwrite(f"{default_path}/lena_smoothed.png", smoothed_image)

def rotation(image, angle):
    a = angle
    if angle == 90:
        angle = cv2.ROTATE_90_CLOCKWISE
    elif angle == 180:
        angle = cv2.ROTATE_180
    elif angle == 270:
        angle = cv2.ROTATE_90_COUNTERCLOCKWISE
    rotated = cv2.rotate(image, angle)
    cv2.imwrite(f"{default_path}/lena_rotated_{a}.png", rotated)

def main():
    img = cv2.imread(f"{default_path}/lena-2.png")
    width, height, channels = img.shape

    padding(img, 100)
    x_1 = 80
    x_2 = width - 130
    y_1 = 80
    y_2 = height - 130
    crop(img, x_1, x_2, y_1, y_2)
    resize(img, 200, 200)

    new_image = np.zeros((width, height, channels), np.uint8)
    copy(img, new_image)
    grayscale(img)
    hsv(img)

    shifted = np.zeros((width, height, channels), np.uint8)
    hue_shifted(img, shifted, 50)

    smoothing(img)
    rotation(img, 90)
    rotation(img, 180)

if __name__ == "__main__":
    default_path = 'assignment_2/images'
    main()