import cv2
import numpy as np


# TLSide = Top left Side Square Length 
# TRSide = Top Right Side Square Length 
# BCSide = Bottom Center Side Square Length 

def cropImage(image,TLSide=300,TRSide=300,BCSide=300):
	mask = np.zeros(image.shape, dtype=np.uint8)
	yMax,xMax,depth = image.shape

	contoursTopLeft = np.array ([[0,0], [TLSide,0], [TLSide,TLSide], [0,TLSide]])
	contoursTopRight = np.array ([[xMax-TRSide,0], [xMax,0], [xMax,TRSide], [xMax-TRSide,TRSide]])


	XCenter = int(xMax/2)
	BCSideHalf = int(BCSide/2)
	print(yMax-BCSide)
	contoursBottomCenter = np.array ([[XCenter-BCSideHalf,yMax-(BCSide)],[XCenter-BCSideHalf,yMax], [XCenter+BCSideHalf,yMax], [XCenter+BCSideHalf,yMax-(BCSide)]])

	#make polygons
	cv2.fillPoly(mask, pts=[contoursTopLeft], color=(255,255,255))
	cv2.fillPoly(mask, pts=[contoursTopRight], color=(255,255,255))
	cv2.fillPoly(mask, pts=[contoursBottomCenter], color=(255,255,255))

	#use polygons to crop & invert image
	masked_image = cv2.bitwise_and(image, mask)
	return masked_image

image = cv2.imread('e3.jpeg')
cv2.imshow("image", cropImage(image,200,250,250))
cv2.waitKey(0)