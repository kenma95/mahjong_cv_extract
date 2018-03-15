# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 16:10:41 2018

@author: kenma
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

input_name = "multiple2.jpg"


def preprocess(image):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    threshold = cv2.adaptiveThreshold(blurred, 255, 1, 1,5, 3)
    threshold = cv2.bitwise_not(threshold)
    return threshold

def crop(img, rect,box):
    mult = 1
    #https://stackoverflow.com/questions/37177811/crop-rectangle-returned-by-minarearect-opencv-python
    W = rect[1][0]
    H = rect[1][1]
    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)
    rotated = False
    angle = rect[2]
    if angle < -45:
        angle+=90
        rotated = True
    center = (int((x1+x2)/2), int((y1+y2)/2))
    size = (int(mult*(x2-x1)),int(mult*(y2-y1)))
    M = cv2.getRotationMatrix2D((size[0]/2, size[1]/2), angle, 1.0)
    cropped = cv2.getRectSubPix(im, size, center)    
    cropped = cv2.warpAffine(cropped, M, size)
    croppedW = W if not rotated else H 
    croppedH = H if not rotated else W
    croppedRotated = cv2.getRectSubPix(cropped, (int(croppedW*mult), int(croppedH*mult)), (size[0]/2, size[1]/2))
    return croppedRotated


def seperate(erosion,im):
    _,contours, hierarchy = cv2.findContours(erosion, cv2.RETR_LIST , cv2.CHAIN_APPROX_TC89_KCOS)
    tiles = []
    for c in contours:
    	perimeter = cv2.arcLength(c, True)
    	approx = cv2.approxPolyDP(c, 0.02*perimeter, True)
    	area = cv2.contourArea(approx)
    	if area > 20000 and area < 500000:
    		tiles.append(approx);

    tiles.sort(key=lambda x: x[0][0][0])
    index = 0
    temp = im.copy()
    temp2 = im.copy()
    canvas = np.zeros((im.shape[0],im.shape[1],3), np.uint8)
    counter=1
    canvas = np.zeros((im.shape[0],im.shape[1],3), np.uint8)
    for tile in tiles:
        rect = cv2.minAreaRect(tile)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        c1 = crop(im, rect,box)
        cv2.drawContours(canvas, [tile], 0, (0, 255, 0), 30)
        #remove unrealistic shape
        if max(c1.shape[0],c1.shape[1])/float(min(c1.shape[0],c1.shape[1])) > 3:
            continue
        plt.imshow(c1)
        cv2.imwrite("segment_%s.jpg" % counter, c1)
        plt.show()
        #draw    
        cv2.drawContours(temp, [tile], 0, (0, 255, 0), 30)
        cv2.drawContours(temp2,[box],0,(0,0,255),30)
        #cv2.imwrite("contour_%s.jpg" % index, temp)
        counter += 1
    plt.imshow(canvas)
    plt.show()
    plt.imshow(temp) 
    plt.show()
    plt.imshow(temp2)
    plt.show()    

    cv2.imwrite("canvas.jpg", canvas)
    cv2.imwrite("rec2.jpg", temp2)
    cv2.imwrite("rec1.jpg", temp)
im = cv2.imread(input_name)

thresh = preprocess(im)
# noise removal
kernel = np.ones((19,19),np.uint8)

erosion = cv2.erode(thresh,kernel,iterations = 3)
plt.imshow(thresh,cmap='binary') 
plt.show()
plt.imshow(erosion,cmap='binary') 
plt.show()
seperate(erosion,im)
