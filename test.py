import cv2
import tree_utils as tu
import numpy as np

def test_TO_HSV():
    #Loading image
    img_bgr = cv2.imread("../FruitLearning/dataset/autumn/568.png")
    
    #converting to hsv
    print(img_bgr.shape)
    img_hsv = tu.to_HSV(img_bgr)
    
    #Displaying
    cv2.namedWindow("Test HSV", cv2.WINDOW_NORMAL)
    cv2.imshow('Test HSV', img_hsv)
    cv2.waitKey(0)

def test_get_luminance():
    #Loading image
    img_bgr = cv2.imread("../FruitLearning/dataset/autumn/568.png")
    
    #Getting the luminance channel
    Y = tu.get_Luminance(img_bgr)
    print(Y.shape)

    #Displaying
    cv2.namedWindow("Test Luminance", cv2.WINDOW_NORMAL)
    cv2.imshow('Test Luminance', Y)
    cv2.waitKey(0)

def test_get_hist():
    #Loading image
    img_bgr = cv2.imread("../FruitLearning/dataset/abricotier_test.jpg")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    #Y = tu.get_Luminance(img_bgr)

    h, h_bins = tu.get_hist(img_rgb)
    #print(h.shape)
    #print(h_bins.shape)
    #print(h_bins)
    #print(h)
    #print(np.sum(h))

def test_get_LBP():
    img_bgr = cv2.imread("../FruitLearning/dataset/abricotier_test.jpg")
    #img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    Y = tu.get_Luminance(img_bgr)
    lbp = tu.get_LBP(Y)
    print(lbp.shape)
    print(np.amax(lbp.flatten()))
    print(lbp)
    print(np.histogram(lbp.flatten(), bins=64))

TO_HSV_TEST = False
GET_LUMINANCE_TEST = False
GET_HIST_TEST = False
GET_LBP_TEST = True

if(TO_HSV_TEST):
    test_TO_HSV()
elif(GET_LUMINANCE_TEST):
    test_get_luminance()
elif(GET_HIST_TEST):
    test_get_hist()
elif(GET_LBP_TEST):
    test_get_LBP()



