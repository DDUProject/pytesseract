import sys
import os
import subprocess
import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('test5.png',0)
rows,cols = img.shape
f=np.array([0]* rows) #one dimentional array to store avarage of each row
k=0;
for i in range(rows):
    for j in range(cols):
        k=k+img[i,j]
f = k/(cols*rows)
#print f

res = cv2.resize(img,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
thr = cv2.adaptiveThreshold(res,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,21,15)
cv2.imshow('display',thr)
cv2.waitKey(0)
cv2.imwrite('try.png',res)
os.system("tesseract try.png test8")
os.remove('try.png')


