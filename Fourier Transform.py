# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 00:55:38 2019

@author: jason
"""
import sys
import time
import math    #math modules (normal and complex) so can use pi, sin etc
import cmath
from scipy import misc
import scipy
import cv2
import numpy as np
import matplotlib.pyplot as plt #plt 用于显示图片

def Fourier(image):
    img = cv2.imread(image)[:,:,0] # gray-scale image
    dim=img.shape #(height, width, channel)
    H, W = dim    
    H_numberli = list(range(H))
    W_numberli = list(range(W))
    new_img = np.zeros((H,W),dtype=complex)
    f_shift = np.zeros((H,W),dtype=complex)

#    Start = time.time() 
    for u in H_numberli:        
        for v in W_numberli:
            Y = np.zeros((1,1))
            for m in H_numberli:        
                for n in W_numberli:
                    X = img[(m),(n)]*(cmath.exp((1j*-2*(math.pi)*(((u*m)/H)+((v*n)/W)))))                       
                    Y = Y + X 
            new_img[(u),(v)] = Y.astype(complex)/(H*W)
#    End = time.time() 
#    print("Total %f sec" % (End - Start))   
    for h in H_numberli:        
        for w in W_numberli:
            if h < (H//2) and w < (W//2) :
                f_shift[((H//2)+h),((W//2)+w)] = new_img[(h),(w)]
            if h < (H//2) and w >= (W//2) :
                f_shift[((H//2)+h),(w-(W//2))] = new_img[(h),(w)]
            if h >= (H//2) and w >= (W//2) :
                f_shift[(h-(H//2)),(w-(W//2))] = new_img[(h),(w)]
            if h >= (H//2) and w < (W//2) :
                f_shift[(h-(H//2)),((W//2)+w)] = new_img[(h),(w)]
    print (np.where(f_shift==0))
    f_abs = np.abs(f_shift) + 1 # lie between 1 and 1e6
    f_bounded = 20 * np.log(f_abs)
    f_bounded = f_bounded - np.min(f_bounded)
    f_img = 255 * f_bounded / np.max(f_bounded)
    f_img = f_img.astype(np.uint8)   
    # plt.imshow(f_img) 
    scipy.misc.imsave('0880304.png', f_img)

Fourier(image=sys.argv[1])