import cv2
import numpy as np
from skimage import feature
import random

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
    # print("shape:", iQ.shape)
    iQ = r + n*g + n*n*b

    maximum_channel = int(np.floor(255.0/n_bins))
    maximum_value = maximum_channel + n*maximum_channel + n*n*maximum_channel
    pas = (maximum_value+1)/n_bins
    bins = [i*pas  for i in range(maximum_value+1)]
    #bins[0] -= 0.001
    #print(len(bins))
    #print("bins",bins)
    #print(np.amax(iQ))
    hist, bin_hist = np.histogram(iQ.flatten(), bins=bins, density=True)
    return np.reshape(hist, (1, len(hist)))

# Get the lbp pattern from the channel passed as argument
# input: image 2D, in this case we are using luminance
# output: histogram with the lbp features
def get_LBP(img):
    assert(len(img.shape) == 2)
    lbp = feature.local_binary_pattern(img, 8, 2, method='ror')
    lbp_hist, lbp_bins = np.histogram(lbp.flatten(), bins=255, density=True)
    return np.reshape(lbp_hist, (1, len(lbp_hist)))

# Find the intersection between two histograms
# input : two histograms
# output : intersection of two histograms : list of minimum elements between each list
def minH(h1i,h2i):
    h1,h2 = h1i,h2i
    h = [h1[i] if h1[i] < h2[i] else h2[i] for i in range(len(h1))]
    return h

# Mesure the similarity between two images by calculating their distance
# input : histogram of each image
# output : distance between the two images
def similarity(h1i,h2i):
    h1,h2 = list(h1i[0]),list(h2i[0])
    assert (len(h1) == len(h2)), "Histogram's lengths do not match"
    sim = 1 - (sum(minH(h1,h2)))/(min(sum(h1),sum(h2)))
    return sim


# Weighted sum of distances between two images
# input : list of signatures for each image and the list of weight for each signature
# output : distance between two images
def distance(signatures1,signatures2,sig_coef):
    assert (len(signatures1) == len(sig_coef) and (len(signatures2) == len(sig_coef)) ), "signatures and coef lengths do not match"
    distances = [-1 for i in range(len(signatures1))]
    # Creation of distance's list for each signature 
    for i in range(len(signatures1)):
        distances[i] = similarity(signatures1[i],signatures2[i])
    # Weighting of distances
    dist = 0
    for i in range(len(distances)):
        dist = dist + distances[i]*sig_coef[i]
    return dist

# Create a list of configurations to use in the learning
# input : total number of images and the k-cross validation number
# output : give a list of indexes composed of k configuration. Each configuration is composed of a training part and a test part
def get_configuration(nb_img,k):
    index = [i for i in range(nb_img)]
    random.shuffle(index)
    nb_int = int(np.floor(nb_img/k))
    train_test = [[] for i in range(k)]
    for i in range(k):
        train_test[i] = [index[nb_int*i:nb_int*(i+1)],index[:nb_int*i]+index[nb_int*(i+1):]]
    return train_test