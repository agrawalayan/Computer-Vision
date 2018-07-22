#import os;
###os.chdir() # Put your own directory
##
##import cv2 
##import numpy as np
##
##image = cv2.imread("TestImages/test1.jpg", cv2.IMREAD_GRAYSCALE)
##
##blurred = cv2.GaussianBlur(image, (5, 5), 0)
##
###blurred  = cv2.bilateralFilter(gray,9,75,75)
##
### apply Canny Edge Detection
##edged = cv2.Canny(blurred, 0, 20)
##
###Find external contour
##
##(im2,contours, _) = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
##
##cv2.imshow("cou", im2)


import cv2

img = cv2.imread("TestImages/test1.jpg")
mser = cv2.MSER_create()

#Resize the image so that MSER can work better
img = cv2.resize(img, (img.shape[1]*2, img.shape[0]*2))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
vis = img.copy()

regions = mser.detectRegions(gray)
hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions[0]]
cv2.polylines(vis, hulls, 1, (0,255,0)) 

cv2.namedWindow('img', 0)
cv2.imshow('img', vis)
cv2.imwrite("segmentation_output.jpg",vis)
