import cv2

#Cchange the colorspace from BGR to HSV
# input: img in the BGR colorspace
# output: img in the HSV colorspace
def to_HSV(img):
    assert(img.shape[2] == 3)
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

# Get only the luminance channel(Y) of the YCrCb space
# input: img in the BGR colorspace
# output: Y channel
def get_Luminance(img):
    assert(img.shape[2] == 3)
    img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(img_YCrCb)
    return Y