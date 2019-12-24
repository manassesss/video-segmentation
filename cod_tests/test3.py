import numpy as np
import argparse
import imutils
import cv2
import time
import pdb

path = 'C:/Users/manas/Documents/PROJETO_SEGMENTAÇÃO_DE_VIDEOS/STRANS/cam10.AVI'
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input',type=str, required=True, help='path to input video')
#ap.add_argument('-m', '--method', type=str, help='which method should be apply in background subtraction (MG2, KNN, O)', default='O')
args = vars(ap.parse_args())

print('[INFO] input :', args['input'])
'''
if args['method'] == 'MG2':
    print('[INFO] selecting the method MG2...')
    bg = cv2.BackgroundSubtractorMOG2()
elif args['method'] == 'KNN':
    print('[INFO] selecting the method KNN...')
    bg = cv2.BackgroundSubtractorKNN()
else:
    print('[INFO] selecting the method simple...')
    bg = cv2.BackgroundSubtractor()
'''
print('[INFO] reading the video...')
cap = cv2.VideoCapture(args['input'])

if not cap.isOpened():
    print('[INFO] the path is not a path to a valid file. verify if the path is correct.')
    exit(0)

while True:
    ret, frame = cap.read()
    if frame is None:
        break
    frame = cv2.cvtColor(frame, cv2.ccccccccccccc
    #print('[INFO] video in gray scale...')
    #fg = bg.apply(frame)

    #print('[INFO] applying subtraction...')
    #cv2.rectangle(frame, (10,2), (100, 20), (255,255,255), -1)
    #cv2.putText(frame, str(cap.get(cv2.CAP_PROP_POS_FRAMES)), (15,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))
    #print('[INFO] exibing the application...')
    cv2.imshow('frame', r)
    #cv2.imshow('mask', fg)
    cv2.waitKey(30)

cap.release()
cv2.destroyAllWindows()