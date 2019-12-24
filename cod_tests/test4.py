import numpy as np
import argparse
import imutils
import cv2
import time
from skimage import exposure
import pdb
from torchvision import models
import torchvision.transforms as T

def roi(frame, vertices):
    mak = np.zeros_like(frame)
    cv2.fillPoly(mak, vertices, 255)
    mask = cv2.bitwise_and(frame, mak)
    return mask

def process_frame(original):
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    frame = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    frame = cv2.medianBlur(frame, 3)
    #frame = cv2.Canny(frame, threshold1=200, threshold2=300)
    vertices = np.array([[0,500],[0,300],[520,100],[550,100],[500,500],], np.int32)
    frame = roi(frame, [vertices])
    return frame

path = 'C:/Users/manas/Documents/PROJETO_SEGMENTAÇÃO_DE_VIDEOS/STRANS/cam02.AVI'
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input',type=str, required=True, help='path to input video')
#ap.add_argument('-m', '--method', type=str, help='which method should be apply in background subtraction (MG2, KNN, O)', default='O')
args = vars(ap.parse_args())


cap = cv2.VideoCapture('strans/cam02.AVI')
frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=50)
frames = []
for fid in frameIds:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = cap.read()
    frames.append(frame)
medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
medianFrame = process_frame(medianFrame)
#grayMedianFrame = exposure.equalize_hist(medianFrame)
#grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)
grayMedianFrame = medianFrame.copy()
cv2.imshow('medianFrame', grayMedianFrame)
cv2.waitKey(0)
subtractor1 = cv2.createBackgroundSubtractorMOG2(10000, 16, 1)
subtractor4 = cv2.createBackgroundSubtractorKNN(dist2Threshold=1000, detectShadows=True)



print('[INFO] input :', args['input'])
print('[INFO] reading the video...')
cap = cv2.VideoCapture(args['input'])
if not cap.isOpened():
    print('[INFO] the path is not a path to a valid file. verify if the path is correct.')
    exit(0)
while True:
    ret, frame = cap.read()
    if frame is None:
        break
    frame = process_frame(frame)
    cv2.imshow('frame', frame)
    dframe = cv2.absdiff(frame, grayMedianFrame)
    cv2.imshow('dframe', dframe)
    mask = subtractor4.apply(dframe)
    cv2.imshow('mask', mask)    
    cv2.waitKey(30)
cap.release()
cv2.destroyAllWindows()