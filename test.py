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


TO_HSV_TEST = False
GET_LUMINANCE_TEST = True

if(TO_HSV_TEST):
    test_TO_HSV()
elif(GET_LUMINANCE_TEST):
    test_get_luminance()    



