"""
Title: Text segmentation
Author: Deep Chakraborty
Date First Created: 26/12/2015
Date Modified: 28/12/2015
"""

import numpy as np
import cv2 
import argparse

# to get the image from command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, 
	help = "Path to the image")
args = vars(ap.parse_args())


image = cv2.imread(args["image"])

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Thresholding
(T, thresh) = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

# keeping only the left half of the pan card as it contains required text
mask = np.zeros(image.shape[:2], dtype = "uint8")
cv2.rectangle(mask, (0,0), (image.shape[1]/2,image.shape[0]), 255, -1)
thresh = cv2.bitwise_and(thresh,mask)

# finding connected areas in the image
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel2, iterations = 1)

# finding contours of text area
(_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Keep only the text areas (that are larger than a certain value)
high = image.copy()
for (i,c) in enumerate(cnts):
	area = cv2.contourArea(c)
	if area > 300: # area limit
		(x,y,w,h) = cv2.boundingRect(c)

		#marking text regions in the original image
		cv2.rectangle(high,(x,y),(x+w,y+h),(255,0,255),2)
		text = image[y:y+h, x:x+w]


		gray = cv2.cvtColor(text, cv2.COLOR_BGR2GRAY)
		# Performing image normalization using Histogram equalization
		equ = cv2.equalizeHist(gray)
		equ = cv2.GaussianBlur(equ, (3,3), 0)

		#Performing adaptive Thresholding
		thresh = cv2.adaptiveThreshold(equ, 255,
			cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 1) 
		
		#displaying all the segmented text areas
		cv2.imshow("text"+str(i), thresh)

		#storing a sample segmented text = Name 
		#works only for image 2
		if i == 19:
			cv2.imwrite("sample.jpg", thresh)
# displaying the original image with highlighted text

# Performing face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = image.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    high = cv2.rectangle(high, (x,y), (x+w,y+h), (255,0,0), 2)
    roi = img[y:y+h, x:x+w]
    # Storing the face image
    cv2.imwrite("Face_" + args["image"], roi)
cv2.imshow("Marked",high)

cv2.waitKey(0)
cv2.destroyAllWindows()