'''
Created on Sep 14, 2015
@author: mattjarvis
'''
# Standard imports
# <NONE>

# Third-party imports
from PIL import Image
import numpy as np
import cv2

# Local imports
# <NONE>

ENCODE_SETTINGS = [int(cv2.IMWRITE_JPEG_QUALITY), 70]


def is_image(path):
    try:
        Image.open(path)
    except IOError:
        return False
    return True


def flip_image(img):
    return cv2.flip(img, 1)


def rotate_image(img):
    # Grab the dimensions of the image
    (h, w) = img.shape[:2]
    # Calculate the center of the image
    center = (w / 2, h / 2)

    # Rotate the image by (0, 90, 180, 270, 360) degrees
    matrix = cv2.getRotationMatrix2D(center, 0, 1.0)
    rotated = cv2.warpAffine(img, matrix, (w, h))
    return rotated


def trim_sides(img):
    dimensions = img.shape    # Get img dimensions (height, width, channels)
    height = dimensions[0]
    width = dimensions[1]

    y = 0                       # Start point: vertical
    h = height                  # End point: vertical
    x = width / 3.5             # Start point: horizontal
    w = width - x               # End point: horizontal

    cropped = img[y:h, x:w]  # Crop
    return cropped


def to_color(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def encode_image(img):
    cv2.imwrite('img.jpg', img, ENCODE_SETTINGS)
    return open('img.jpg')


def to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def to_colour(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def crop_image(img, area):
    (x, y, w, h) = (area[0], area[1], area[2], area[3])
    return img[y: y + h, x: x + w]


def image_to_numpy(img):
    np_img = np.array(img, 'uint8')
    return np_img


def image_list_to_numpy_list(imgs):
    np_imgs = []
    for img in imgs:
        np_imgs.append(image_to_numpy(img))
    return np_imgs


def numpy_to_image(np_img):
    img = Image.fromarray(np_img, 'RGB')
    return img


def cvt_images_to_gray_np(img_list):
    imgs = []
    for img in img_list:
        gray_img = to_grayscale(img)
        np_img = image_to_numpy(gray_img)
        imgs.append(np_img)
    return imgs
