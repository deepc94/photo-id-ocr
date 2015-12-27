"""
Title: Image Preprocessing
Author: Deep Chakraborty
Date First Created: 26/12/2015
Date Modified: 28/12/2015

The code currently has many redundancies that need to be fixed
"""


import cv2
import numpy as np
import argparse
from math import atan

# to get the image from command line arguments

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, 
	help = "Path to the image")
args = vars(ap.parse_args())


image = cv2.imread(args["image"])

# Part 1: Extracting only colored regions from the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(image, (3,3), 0)
(T, thresh) = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

# kernel = np.ones((5,5),np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 5)

(_, cnts, _) = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
       cv2.CHAIN_APPROX_SIMPLE)
(x, y, w, h) = cv2.boundingRect(cnts[0])
cropped = image[y:y+h, x:x+w]

# Part 2: De-skewing the image

image = cropped.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(image, (5,5), 0)

canny = cv2.Canny(image, 30, 150)
kernel = np.ones((2,2),np.uint8)
(_, cnts, _) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL,
       cv2.CHAIN_APPROX_SIMPLE)
im = image.copy()

rect = cv2.minAreaRect(cnts[-1]) # using the largest contour in the image
box = cv2.boxPoints(rect)
box = np.int0(box)
im = cv2.drawContours(im,[box],0,(0,0,255),2)
mask = np.zeros(image.shape[:2], dtype = "uint8")
cv2.fillConvexPoly(mask, box, (255, 255, 255))
masked = cv2.bitwise_and(image, image, mask = mask)

x0,y0 = box[0]
x1,y1 = box[1]
x2,y2 = box[2]

# calculation of skew angle

angle = 90.0 - (atan(float(x0-x1)/(y0-y1)) * 180/3.14)
center = (((x0+x2)//2),((y0+y2)//2))
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(masked, M, (image.shape[1], image.shape[0])) # de-skewing,
# can otherwise be performed using warpPerspective() and 4-point Transform


# Part 3: Cropping the ROI

image = rotated.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(image, (3,3), 0)
(T, thresh) = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

kernel = np.ones((5,5),np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

(_, cnts, _) = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
       cv2.CHAIN_APPROX_SIMPLE)
(x, y, w, h) = cv2.boundingRect(cnts[0])
cropped = image[y:y+h, x:x+w]
cv2.imshow("Cropped", cropped)
cv2.imwrite("f_" + args["image"],cropped)


cv2.waitKey(0)