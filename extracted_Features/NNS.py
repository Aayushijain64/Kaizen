import numpy as np
import math
import operator
def getDistance(a,b):
	x1=a[0]
	y1=a[1]
	x2=b[0]
	y2=b[1]
	dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
	return dist

# Input - 
# a,b,c are the coordinates of 3 triradii found after training
# Arr is the array of points (2 length arrays like [0,2],[0,0]) 

# Output - 
# Array containing three lists [closest1,closest2,closest3]
# closest1 = [point1,point2,point3,...]
# In closest1, there are sorted points based on distances from point a
# point1 will be closest to a, and so on

# Similary for closest2,closest3
# closest2 = [point1,point2,point3,...]
# closest3 = [point1,point2,point3,...]

# For your purpose aayushi, you can just take 
# closest1[0],closest2[0],closest3[0] as the three points
# from which you take out the atd angle.

def findNearest(a,b,c,Arr):
	distancesArray1=[]
	indexArray1=[]
	distancesArray2=[]
	indexArray2=[]
	distancesArray3=[]
	indexArray3=[]

	for i in range(0,len(Arr)):
		distancesArray1.append(getDistance(Arr[i],a))
		indexArray1.append(i)
		distancesArray2.append(getDistance(Arr[i],b))
		indexArray2.append(i)
		distancesArray3.append(getDistance(Arr[i],c))
		indexArray3.append(i)

	closest1=[k for k, v in sorted(zip(indexArray1,distancesArray1), key=operator.itemgetter(1))]
	closest2=[k for k, v in sorted(zip(indexArray2,distancesArray2), key=operator.itemgetter(1))]
	closest3=[k for k, v in sorted(zip(indexArray3,distancesArray3), key=operator.itemgetter(1))]
	
	for i in range(0,len(Arr)):
		closest1[i]=Arr[closest1[i]]
		closest2[i]=Arr[closest2[i]]
		closest3[i]=Arr[closest3[i]]
	return [closest1,closest2,closest3]

def angle(pt1,pt2,pt3):
	P12=getDistance(pt2,pt1)
	P23=getDistance(pt1,pt3)
	P13=getDistance(pt2,pt3)
	return np.arccos((P12**2 + P13**2 - P23**2) / (2*P12*P13))*180/np.pi

a=[1,1]
b=[0,0]
c=[-1,1]

print(angle(a,b,c))