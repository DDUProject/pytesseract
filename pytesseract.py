import cv2
import os
import sys
import numpy as np
from SkewDetectAndCorrect import SkewDetectAndCorrect
import pytesser as pt

#different parameters to set
lang = "eng"
output = None
image = None
thr = False     #if thr true then do adaptive thresholding
psm = 3         #default psm is PSM_AUTO
arg = 1
argc = len(sys.argv)

while(arg<argc and (output==None or sys.argv[arg][0] == '-')):
    if((sys.argv[arg] == "-l") and (arg+1 < argc)):
        lang = sys.argv[arg+1]
        arg = arg+1
    elif((sys.argv[arg] == "-psm") and (arg+1 < argc)):
        psm = sys.argv[arg+1]
        arg = arg+1
    elif((sys.argv[arg] == "-thr") and (arg+1 < argc)):
        if(sys.argv[arg+1] == "adap"):
            thr=True
        arg = arg+1
    elif(image==None):
        image = sys.argv[arg]
    elif(output==None):
        output = sys.argv[arg]
    arg=arg+1



img = cv2.imread(image,0)         #read image for pre-processing
#img = cv2.imread('skew4.jpg',0)
img = cv2.resize(img,None,fx=2.5,fy=2.5,interpolation=cv2.INTER_CUBIC)  #interpolate for better detection
img_deskewed,img_edges = SkewDetectAndCorrect(img)      #correct skew in image if it is there
img_deskewed_blured = cv2.medianBlur(img_deskewed,3)    # low pass filter for reducing noise

rows,cols = img_deskewed.shape      #find size
maxValueOfRowsOfEdgesImage=[0]      #defined array with first element 0

# finds maximum values from rows of edges image and store
for i in range(rows):
        maxValueOfRowsOfEdgesImage.append(max(img_edges[i,:]))
maxValueOfRowsOfEdgesImage.append(0)    #append last element 0 for reducing errors

maxValueOfRowsOfEdgesImage=np.array(maxValueOfRowsOfEdgesImage)     #convert list into array

upper_line=[]   #stores elements from maxValueOfRowsOfEdgesImage which denotes starting of line in image
lower_line=[]   #stores elements from maxValueOfRowsOfEdgesImage which denotes ending of line in image

#this loop finds  at which row in image, line starts and at which row line in image ends
for i in range(rows+1):    
    if ((maxValueOfRowsOfEdgesImage[i+1]==0)|(maxValueOfRowsOfEdgesImage[i]==0)):
        if ((maxValueOfRowsOfEdgesImage[i+1]!=0) & (maxValueOfRowsOfEdgesImage[i]==0)):
            #img_deskewed = cv2.line(img_deskewed,(0,i),(cols,i),(100,0,100),1) #this line draws upper line on image
            upper_line.append(i)
        elif((maxValueOfRowsOfEdgesImage[i+1]==0) & (maxValueOfRowsOfEdgesImage[i]!=0)):
            #img_deskewed = cv2.line(img_deskewed,(0,i),(cols,i),(100,0,100),1) #this line draws lower line on image
            lower_line.append(i)

upper_line=np.array(upper_line)     #converts list in array
lower_line=np.array(lower_line)


#cuts each line from image and does character recognition for each lines separately
fout = open(output+".txt","w")
print "Please wait while I'm processing your Image..."
for i in range(len(upper_line)):
        h = lower_line[i]-upper_line[i]     #hight of line
        y = upper_line[i]                   #Y coordinate of line
        img_line = img_deskewed[y:y+h,0:cols]   #cut detected line from image and save in img_line
        if(thr == True):
            img_line = cv2.adaptiveThreshold(img_line,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
               cv2.THRESH_BINARY,15,2)
        txt = pt.iplimage_to_string(img_line, lang, psm)   #txt contains recognized string from line
        fout.write(txt)
fout.close()
