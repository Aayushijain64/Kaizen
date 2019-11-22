import cv2
import numpy as np
import skimage.morphology
import skimage
import utils
import math
import argparse
import os
from PIL import Image, ImageDraw
from skimage.morphology import convex_hull_image, erosion
from skimage.morphology import square

def removeSpuriousMinutiae(minutiaeList, img, thresh):
    img = img * 0;
    SpuriousMin = [];
    numPoints = len(minutiaeList);
    D = np.zeros((numPoints, numPoints))
    for i in range(1,numPoints):
        for j in range(0, i):
            (X1,Y1) = minutiaeList[i]['centroid']
            (X2,Y2) = minutiaeList[j]['centroid']
            
            dist = np.sqrt((X2-X1)**2 + (Y2-Y1)**2);
            D[i][j] = dist
            if(dist < thresh):
                SpuriousMin.append(i)
                SpuriousMin.append(j)
                
    SpuriousMin = np.unique(SpuriousMin)
    for i in range(0,numPoints):
        if(not i in SpuriousMin):
            (X,Y) = np.int16(minutiaeList[i]['centroid']);
            img[X,Y] = 1;
    
    img = np.uint8(img);
    return(img)
def getTerminationBifurcation(img, mask):
    img = img == 255;
    (rows, cols) = img.shape;
    minutiaeTerm = np.zeros(img.shape);
    minutiaeBif = np.zeros(img.shape);
    
    for i in range(1,rows-1):
        for j in range(1,cols-1):
            if(img[i][j] == 1):
                block = img[i-1:i+2,j-1:j+2];
                block_val = np.sum(block);
                if(block_val == 2):
                    minutiaeTerm[i,j] = 1;
                elif(block_val == 4):
                    minutiaeBif[i,j] = 1;
    
    mask = convex_hull_image(mask>0)
    mask = erosion(mask, square(5))        
    minutiaeTerm = np.uint8(mask)*minutiaeTerm
    return(minutiaeTerm, minutiaeBif)
signum = lambda x: -1 if x < 0 else 1
cells = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
original = cv2.imread("palm2.jpeg") 
cv2.imshow("palm resized",original) #to view the palm in python
cv2.waitKey(0)
cv2.destroyAllWindows()
img = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
cv2.imshow("palm grayed",img) #to view the palm in python
cv2.waitKey(0)
cv2.destroyAllWindows()
img = cv2.equalizeHist(img)
cv2.imshow("equalized",img) #to view the palm in python
cv2.waitKey(0)
cv2.destroyAllWindows()

img = cv2.GaussianBlur(img, (9, 9), 0)
cv2.imshow("blurred",img) #to view the palm in python
cv2.waitKey(0)
cv2.destroyAllWindows()

img = cv2.Canny(img, 170, 190)
cv2.imshow("canny",img) #to view the palm in python
cv2.imwrite('c.jpeg',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
img = np.uint8(img>128);
skel = skimage.morphology.skeletonize(img)
skel = np.uint8(skel)*255;
mask = img*255;
(minutiaeTerm, minutiaeBif) = getTerminationBifurcation(skel, mask);
minutiaeTerm = skimage.measure.label(minutiaeTerm, 8);
RP = skimage.measure.regionprops(minutiaeTerm)
minutiaeTerm = removeSpuriousMinutiae(RP, np.uint8(img), 10);
BifLabel = skimage.measure.label(minutiaeBif, 8);
TermLabel = skimage.measure.label(minutiaeTerm, 8);
minutiaeBif = minutiaeBif * 0;
minutiaeTerm = minutiaeTerm * 0;
(rows, cols) = skel.shape
DispImg = np.zeros((rows,cols,3), np.uint8)
DispImg[:,:,0] = skel; DispImg[:,:,1] = skel; DispImg[:,:,2] = skel;
RP = skimage.measure.regionprops(TermLabel)
for i in RP:
    (row, col) = np.int16(np.round(i['Centroid']))
    minutiaeTerm[row, col] = 1;
    (rr, cc) = skimage.draw.circle_perimeter(row, col, 50);
    skimage.draw.set_color(DispImg, (rr,cc), (0, 0, 255));
cv2.imshow('a',DispImg);
cv2.imwrite('a.jpeg',DispImg)
cv2.waitKey(0)
cv2.destroyAllWindows()

lined = np.copy(original) * 0
lines = cv2.HoughLinesP(img, 1, np.pi / 180, 15, np.array([]), 50, 20)
for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(lined, (x1, y1), (x2, y2), (0, 255, 0))
cv2.imshow("Line detected",lined) #to view the palm in python
cv2.waitKey(0)
cv2.destroyAllWindows()

output = cv2.addWeighted(original, 0.8, DispImg, 1, 0)
cv2.imshow("output",output) #to view the palm in python
cv2.waitKey(0)
cv2.destroyAllWindows()
width = 250
height = 300
dim = (width, height)
resized = cv2.resize(output, dim, interpolation = cv2.INTER_AREA)
cv2.imshow("palm",resized) #to view the palm in python
cv2.waitKey(0)
cv2.imwrite('b.jpeg',output)
cv2.destroyAllWindows()
