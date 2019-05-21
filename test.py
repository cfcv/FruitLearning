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
    img_bgr = cv2.imread("../FruitLearning/Resources/database/autumn/481.png")
    #img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    Y = tu.get_Luminance(img_bgr)
    lbp = tu.get_LBP(Y)
    print(lbp.shape)
    print(np.amax(lbp.flatten()))
    print(lbp)
    print(np.histogram(lbp.flatten(), bins=64))


def test_distance():
    img_abr = cv2.imread("../FruitLearning/Resources/test/abricotier_test.jpg")
    img_sum = cv2.imread("../FruitLearning/Resources/test/summer_test.jpg")
    img_aut = cv2.imread("../FruitLearning/Resources/test/autumn_test.jpg")

    lum_abr = tu.get_Luminance(img_abr)
    lum_sum = tu.get_Luminance(img_sum)
    lum_aut = tu.get_Luminance(img_aut)

    # LBP
    lbp_abr = tu.get_LBP(lum_abr)
    lbp_sum = tu.get_LBP(lum_sum)
    lbp_aut = tu.get_LBP(lum_aut)

    # sim_abr_abr = tu.similarity(lbp_abr,lbp_abr)
    # sim_abr_sum = tu.similarity(lbp_abr,lbp_sum)
    # sim_abr_aut = tu.similarity(lbp_abr,lbp_aut)
    # sim_aut_sum = tu.similarity(lbp_aut,lbp_sum)

    # Histogram
    h_abr = tu.get_hist(img_abr)
    h_sum = tu.get_hist(img_sum)
    h_aut = tu.get_hist(img_aut)


    print(" --------------- Distance : ---------------")

    sig_abr = [lbp_abr,h_abr]
    sig_aut = [lbp_aut,h_aut]
    sig_sum = [lbp_sum,h_sum]
    coef_sig = [0.2,0.8]

    dist_abr_sum = tu.distance(sig_abr,sig_sum,coef_sig)
    dist_abr_aut = tu.distance(sig_abr,sig_aut,coef_sig)
    dist_sum_aut = tu.distance(sig_sum,sig_aut,coef_sig)

    print("Distance abr_sum", dist_abr_sum)
    print("Distance abr_aut", dist_abr_aut)
    print("Distance sum_aut", dist_sum_aut)


def test_get_k_cross_config(img_size,k):
    config = tu.get_k_cross_configuration(img_size,k)
    print("Get k-cross configuration test : img_size = ",img_size," k = ",k)
    print(config)

def test_get_NN_config(img_size):
    config = tu.get_nn_configuration(img_size)
    print("Get NN configuration test : img_size = ",img_size)
    print(config)



TO_HSV_TEST = False
GET_LUMINANCE_TEST = False
GET_HIST_TEST = False
GET_LBP_TEST = False
DISTANCE_TEST = False
GET_K_CROSS_CONFIG_TEST = False
GET_NN_CONFIG_TEST = True


if(TO_HSV_TEST):
    test_TO_HSV()
elif(GET_LUMINANCE_TEST):
    test_get_luminance()
elif(GET_HIST_TEST):
    test_get_hist()
elif(GET_LBP_TEST):
    test_get_LBP()
elif(DISTANCE_TEST):
    test_distance()
elif(GET_K_CROSS_CONFIG_TEST):
    img_size, k = 910,910 
    test_get_k_cross_config(img_size,k)
elif(GET_NN_CONFIG_TEST):
    img_size = 910
    test_get_NN_config(img_size)


