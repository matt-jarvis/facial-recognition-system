"""
Created on Sep 7, 2015

@author: mattjarvis

TODO:   open_cam/close
"""
# Standard imports
# <NONE>

# Third-party imports
import cv2

# Local imports
# <NONE>

FRAME_RATE = 25
SLEEP = 1.0 / FRAME_RATE
cam = cv2.VideoCapture(0)


def get_raw_image():
    return cam.read()
