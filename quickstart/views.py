from django.contrib.auth.models import User, Group
from quickstart.models import Language
from rest_framework import viewsets
from quickstart.serializers import UserSerializer, GroupSerializer, LanguageSerializer

from django.shortcuts import render
from django.http import Http404
from rest_framework.views import APIView
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from django.core import serializers
from django.conf import settings
import json
from sklearn import tree
import cv2
import numpy as np
import base64
# from final import extractFeatures

@api_view(["POST"])
def GetHypertensionWeight(request):
    r=request
    files=r.FILES.dict()
    leftHandImageBytes = files['left'].read()
    rightHandImageBytes = files['right'].read()
    # print(type(leftHandImage))
    leftNpArray = np.fromstring(leftHandImageBytes, np.uint8)
    rightNpArray = np.fromstring(rightHandImageBytes, np.uint8)

    leftHandImage = cv2.imdecode(leftNpArray, cv2.IMREAD_COLOR)
    rightHandImage = cv2.imdecode(rightNpArray, cv2.IMREAD_COLOR)

    cv2.imshow('',rightHandImage)
    cv2.waitKey(0)

    try:
        risk = parseImages(leftHandImage,rightHandImage)[0]*100
        # if risk==0:
        #     res='Yes, you are at risk of hypertension'
        # else:
        #     res='No, you are not at risk of hypertension'
        # weight=str(height*10)
        return JsonResponse("HyperTension Risk - "+str(risk)+"%",safe=False) 
        # return JsonResponse("Ideal weight should be:"+weight+" kg",safe=False)
    except ValueError as e:
        return Response(e.args[0],status.HTTP_400_BAD_REQUEST)

def parseImages(leftHandImage,rightHandImage):
    return getDisease(1,2,3,4)

def getDisease(radialLoops,ulnarLoops,arches,whorls):   #add theta later
    # features = [[1.016,3.84,0.54,4.58],[0.76,5.98,0.71,2.53]]
    label=[1,0]

    radialFeatures = [[1.016],[0.76]]
    # radialLabels = [1,0]

    ulnarFeatures = [[3.84],[5.98]]
    # ulnarLabels = [1,0]

    archesFeatures = [[0.54],[0.71]]
    # archesLabels = [1,0]

    whorlFeatures = [[4.58],[2.53]]
    # whorlLabels = [1,0]

    # labels = [1,0]  #1 denotes yes and 2 denotes no for hypertension risk
    decisionMakerRadial = tree.DecisionTreeClassifier()
    decisionMakerRadial.fit(radialFeatures,label)

    decisionMakerUlnar = tree.DecisionTreeClassifier()
    decisionMakerUlnar.fit(ulnarFeatures,label)
    
    decisionMakerArches = tree.DecisionTreeClassifier()
    decisionMakerArches.fit(archesFeatures,label)
    
    decisionMakerWhorls = tree.DecisionTreeClassifier()
    decisionMakerWhorls.fit(whorlFeatures,label)

    # clf = clf.fit(features,labels)

    r = decisionMakerRadial.predict([[radialLoops]])
    u = decisionMakerUlnar.predict([[ulnarLoops]])
    a = decisionMakerArches.predict([[arches]])
    w = decisionMakerWhorls.predict([[whorls]])

    print('r,u,a,w=',r,u,a,w)

    return (r+u+a+w)/4

class UserViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows users to be viewed or edited.
    """
    queryset = User.objects.all().order_by('-date_joined')
    serializer_class = UserSerializer


class GroupViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows groups to be viewed or edited.
    """
    queryset = Group.objects.all()
    serializer_class = GroupSerializer

class LanguageViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows groups to be viewed or edited.
    """
    queryset = Language.objects.all()
    serializer_class = LanguageSerializer
