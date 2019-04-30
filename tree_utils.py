import cv2

def to_HSV(img):
    assert(img.shape[2] == 3)
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)