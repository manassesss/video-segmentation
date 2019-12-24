import numpy as np
import cv2

from skimage import filters, morphology

def filter_mask(img, a=None):
        '''
            This filters are hand-picked just based on visual tests
        '''

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

        # Fill any small holes
        #closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        # Remove noise
        #opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) 
        # Dilate to merge adjacent blobs
        #dilation = cv2.dilate(img, kernel)
        median = cv2.medianBlur(img, 3)


        return median


# Open Video
cap = cv2.VideoCapture('strans/cam10.AVI')
 
# Randomly select 25 frames
frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=50)
 
# Store selected frames in an array
frames = []
for fid in frameIds:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = cap.read()
    frames.append(frame)
 
# Calculate the median along the time axis
medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)
 
# Display median frame
cv2.imshow('medianFrame', medianFrame)
cv2.waitKey(0)
# Reset frame number to 0
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
 
# Convert background to grayscale
grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)


#subtractors
subtractor1 = cv2.createBackgroundSubtractorMOG2(1900, 16, 1)
subtractor4 = cv2.createBackgroundSubtractorKNN(dist2Threshold=1000, detectShadows=True)


# Loop over all frames
ret = True
while(ret):
  # Read frame
  ret, frame = cap.read()
  # Convert current frame to grayscale
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  cv2.imshow('original', frame)
  # Calculate absolute difference of current frame and the median frame
  dframe = cv2.absdiff(frame, grayMedianFrame)
  cv2.imshow('absolute', dframe)

  mask = subtractor4.apply(dframe)
  cv2.imshow('mask', mask)
  # Treshold to binarize
  th, dframe = cv2.threshold(mask, 150, 220, cv2.THRESH_BINARY)

  # Display image
  cv2.imshow('frame', dframe)
  cv2.waitKey(20)


# Release video object
cap.release()
cv2.waitKey()
# Destroy all windows
cv2.destroyAllWindows()