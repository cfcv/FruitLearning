import cv2
import numpy as np
from skimage import feature

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

def get_hist(img):
    #assert(len(img.shape) == 2)
    n = 4
    n_bins = n**3
    
    r, g, b = cv2.split(img)
    
    r = np.floor(r.flatten() / n_bins)
    g = np.floor(g.flatten() / n_bins)
    b = np.floor(b.flatten() / n_bins)

    iQ = np.zeros(r.shape)
    #iQ = np.floor(iQ/n_bins)
    print("shape:", iQ.shape)
    iQ = r + n*g + n*n*b

    maximum_channel = int(np.floor(255.0/n_bins))
    maximum_value = maximum_channel + n*maximum_channel + n*n*maximum_channel
    pas = (maximum_value+1)/n_bins
    bins = [i*pas  for i in range(maximum_value+1)]
    #bins[0] -= 0.001
    #print(len(bins))
    #print("bins",bins)
    #print(np.amax(iQ))
    return np.histogram(iQ, bins=bins, normalized=True)

def get_LBP(img):
    assert(len(img.shape) == 2)
    return feature.local_binary_pattern(img, 8, 2, method='ror')