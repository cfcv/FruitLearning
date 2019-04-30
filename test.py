import cv2
import tree_utils as tu

def test_TO_HSV():
    #Loading image
    img_bgr = cv2.imread("/home/cfcv/Desktop/FruitLearning/dataset/autumn/568.png")
    
    #converting to hsv
    print(img_bgr.shape)
    
    #Displaying
    cv2.namedWindow("Test HSV", cv2.WINDOW_NORMAL)
    cv2.imshow('Test HSV', tu.to_HSV(img_bgr))
    cv2.waitKey(0)

TO_HSV_TEST = True

if(TO_HSV_TEST):
    test_TO_HSV()



