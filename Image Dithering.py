# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 01:02:36 2019

@author: jason
"""

import matplotlib.pyplot as plt #plt 用于显示图片
import matplotlib.image as mpimg #mpimg 用于读取图片
import numpy as np
from scipy import misc
import scipy
import array
import sys

def Ordered_Dither(N=8,image='input_image'):    
    if N == 2:  
        I=np.array([[1,2],[3,0]])
    if N == 4:      
        I=np.array([[5,9,6,10],[13,1,14,2],[7,11,4,8],[15,3,12,0]]).reshape((4,4))
    if N == 8:  
        I=np.array([[62,57,48,36,37,49,58,63],[56,47,35,21,22,38,50,59],
                      [46,34,20,10,11,23,39,51],[33,19,9,3,0,4,12,24],
                      [32,18,8,2,1,5,13,25],[45,31,17,7,6,14,26,40],
                      [55,44,30,16,15,27,41,52],[61,54,43,29,28,42,53,60]])      
    img = scipy.misc.imread(str(image)) 
    dim=img.shape #(height, width, channel)
    def rgb2gray(rgb):return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    if len(dim) != 2:
        gray = rgb2gray(img)
        H, W, c = dim
    else:
        gray = img
        H, W = dim
    H_numberli = list(range(H))
    W_numberli = list(range(W))
    new_picture = np.zeros((H,W))
    T=255*((I+0.5)/(N*N))
    for h_index in H_numberli:        
        for w_index in W_numberli:
            X = gray[(h_index),(w_index)] 
            if X > T[(h_index%N),(w_index%N)] :
                X = 255
            else: X = 0       
            new_picture[(h_index),(w_index)] = X
    # plt.imshow(new_picture) # 显示图片
    scipy.misc.imsave('0880304.png', new_picture)

Ordered_Dither(image=sys.argv[1])