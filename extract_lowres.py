# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 16:10:41 2018

Deprecated
This is a develop version to run the extraction on low res(640*480)
tile image. Still functional but not the latest

@author: kenma
"""


import numpy as np
import cv2
from matplotlib import pyplot as plt

def preprocess(image):
	gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (7, 7), 0)
	threshold = cv2.adaptiveThreshold(blurred, 255, 1, 1, 3, 5)
	return threshold


def crop(img, rect):
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

im = cv2.imread('multiple2_l.jpg')

thresh = preprocess(im)
thresh = cv2.bitwise_not(thresh)

# noise removal
kernel = np.ones((7,7),np.uint8)

#dilation = cv2.dilate(thresh,kernel,iterations = 1)
erosion = cv2.erode(thresh,kernel,iterations = 3)
plt.imshow(thresh,cmap='binary') 
plt.show()
plt.imshow(erosion,cmap='binary') 
plt.show()
_,contours, hierarchy = cv2.findContours(erosion, cv2.RETR_LIST , cv2.CHAIN_APPROX_TC89_KCOS)
tiles = []
for c in contours:
	perimeter = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02*perimeter, True)
	area = cv2.contourArea(approx)
	if area > 1000 and area < 10000:
		tiles.append(approx);

tiles.sort(key=lambda x: x[0][0][0])
index = 0
temp = im.copy()
temp2 = im.copy()
canvas = np.zeros((im.shape[0],im.shape[1],3), np.uint8)
counter=1
for tile in tiles:
    
    canvas = np.zeros((im.shape[0],im.shape[1],3), np.uint8)
    cv2.drawContours(temp, [tile], 0, (0, 255, 0), 3)
    cv2.drawContours(canvas, [tile], 0, (0, 255, 0), 3)
    rect = cv2.minAreaRect(tile)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    c1 = crop(im, rect)
    plt.imshow(c1)
    cv2.imwrite("contour_%s.jpg" % index, temp)
    
    plt.show()

    cv2.drawContours(temp2,[box],0,(0,0,255),2)
    plt.imshow(temp,cmap='binary') 
    plt.show()
    plt.imshow(temp2,cmap='binary') 
    plt.show()
    cv2.imwrite("contour_%s.jpg" % index, temp)
    index += 1
def inverte(imagem, name):
    imagem = (255-imagem)
    cv2.imwrite(name, imagem)

inverte(thresh,"thresh.jpg")
save_binary(erosion,"erosion")