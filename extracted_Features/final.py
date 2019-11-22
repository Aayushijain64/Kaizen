import cv2
import numpy as np
import skimage.morphology
import skimage
import utils
import math
# import argparse
# import os
from PIL import Image, ImageDraw
from skimage.morphology import convex_hull_image, erosion
from skimage.morphology import square
from urllib.request import urlopen
# import imutils

def url_to_image(url, readFlag=cv2.IMREAD_COLOR):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    # resp = urlopen(url)
    # image = np.asarray(bytearray(resp.read()), dtype="uint8")
    # image = cv2.imdecode(image, readFlag)
    # print(image)
    # image = image.convert('L')
    # image = np.asarray(gray)
    # return the image

    cap = cv2.VideoCapture(url)
    if( cap.isOpened() ) :
        ret,img = cap.read()
        # cv2.imshow("win",img)
        # cv2.waitKey()
        return img

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
def get_angle(left, right):
    angle = left - right
    if abs(angle) > 180:
        angle = -1 * signum(angle) * (360 - abs(angle))
    return angle

def poincare_index_at(i, j, angles, tolerance):
    deg_angles = [math.degrees(angles[i - k][j - l]) % 180 for k, l in cells]
    index = 0
    for k in range(0, 8):
        if abs(get_angle(deg_angles[k], deg_angles[k + 1])) > 90:
            deg_angles[k + 1] += 180
        index += get_angle(deg_angles[k], deg_angles[k + 1])

    if 180 - tolerance <= index and index <= 180 + tolerance:
        return "loop"
    if -180 - tolerance <= index and index <= -180 + tolerance:
        return "delta"
    if 360 - tolerance <= index and index <= 360 + tolerance:
        return "whorl"
    return "none"

def calculate_singularities(im, angles, tolerance, W):
    (x, y) = im.size
    result = im.convert("RGB")

    draw = ImageDraw.Draw(result)

    colors = {"loop" : (150, 0, 0), "delta" : (0, 150, 0), "whorl": (0, 0, 150)}

    for i in range(1, len(angles) - 1):
        for j in range(1, len(angles[i]) - 1):
            singularity = poincare_index_at(i, j, angles, tolerance)
            if singularity != "none":
                draw.ellipse([(i * W, j * W), ((i + 1) * W, (j + 1) * W)], outline = colors[singularity])

    del draw

    return result

if __name__ == "__main__":
# def extractFeatures(url):  
    
    img = cv2.imread('./e1.jpg',0);
    # img = url_to_image('https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcT0zXkDx9KUYOscO8mPDjRrraQNwU1gBcynO8-8b07DGYww3TXF',0);
    # img=img[:,:,0]
    img = ~img
    cv2.imshow('a',img);
    cv2.waitKey(0)
    print(img.shape)
    
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
    
    RP = skimage.measure.regionprops(BifLabel)
    for i in RP:
        (row, col) = np.int16(np.round(i['Centroid']))
        minutiaeBif[row, col] = 1;
        (rr, cc) = skimage.draw.circle_perimeter(row, col, 3);
        skimage.draw.set_color(DispImg, (rr,cc), (255,0,0));
    
    RP = skimage.measure.regionprops(TermLabel)
    for i in RP:
        (row, col) = np.int16(np.round(i['Centroid']))
        minutiaeTerm[row, col] = 1;
        (rr, cc) = skimage.draw.circle_perimeter(row, col, 3);
        skimage.draw.set_color(DispImg, (rr,cc), (0, 0, 255));
    
    cv2.imshow('a',DispImg);
    cv2.waitKey(0)
    im = Image.open('./e.jpeg')
    im = im.convert("L")  # covert to grayscale
    W = int(20)
    f = lambda x, y: 2 * x * y
    g = lambda x, y: x ** 2 - y ** 2
    angles = utils.calculate_angles(im, W, f, g)
    result = calculate_singularities(im, angles, int(5), W)
    result.show()
    # cv2.destroyallwindows()
