# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 01:10:14 2019

@author: jason
"""
import sys
import math    
import cv2

import numpy as np
import matplotlib.pyplot as plt #plt 用于显示图片
#################################################################################################
#################################################################################################
def dilation(image,SE,R): 
    img = image
    width = image.shape[1]
    height = image.shape[0]   
    pad_image = np.zeros((height+R*2,width+R*2),dtype=int)
    pad_image[R:-R,R:-R] = image    
    image = pad_image
    D = image.copy()
    for i in range(0,height):
        for j in range(0,width):
            ao =[]
            for r1 in range(0,R):
                for r2 in range(0,R):
                    a = image[i-r1,j-r2]
                    S_E = SE[r1,r2]
                    ao.append(a & S_E)      
            D[i,j]=max(ao);img=D[R:-R,R:-R]
    cv2.imwrite('0880304_dilation.png',img*255)       
#    return img
#plt.imshow(img)
#################################################################################################
def erosion(image,SE,R):  
    E = image.copy()
    width = image.shape[1]
    height = image.shape[0]
    for i in range(0,height):
        for j in range(0,width):
            ao =[]
            for r1 in range(0,R):
                for r2 in range(0,R):
                    a = image[i-r1,j-r2]
                    S_E = SE[r1,r2]
                    if (S_E==1 and a==S_E):
                        ao.append(1)
                    else:
                        ao.append(0)
            E[i,j]=min(ao)  
    cv2.imwrite('0880304_erosion.png',E*255)       
#    return E
#plt.imshow(E)
#################################################################################################
image = cv2.imread(str(sys.argv[1]))[:,:,0]
SE = cv2.imread(str(sys.argv[2]))[:,:,0]
R = int(sys.argv[3])
dilation(image,SE,R)
erosion(image,SE,R)
#################################################################################################