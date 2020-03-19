import cv2
import numpy as np
from Filtering import Filtering as f
from skimage import filters, morphology
from PIL import Image

# Create a VideoCapture object
cap = cv2.VideoCapture('strans/cam20.AVI')
 
# Check if camera opened successfully
if (cap.isOpened() == False): 
  print("Unable to read camera feed")
 
# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
 
# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
#out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height), 0)
 
#subtractors
subtractor1 = cv2.createBackgroundSubtractorMOG2(800, 16, True)
subtractor4 = cv2.createBackgroundSubtractorKNN(dist2Threshold=1000, detectShadows=True)
frames = []
# Loop over all frames
alt = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
comp = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(alt, comp)
outputname = 'C:\\Users\\manas\\Documents\\PROJETO_SEGMENTAÇÃO_DE_VIDEOS\\results\\out5-2.avi'
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
resolution = (2*comp, alt)
out = cv2.VideoWriter(outputname, fourcc, 10, resolution, 0)
kernel = np.ones((3,3), np.uint8)

while(True):
  ret, frame = cap.read()
  cont = 200
  if ret == True:
    hv = cv2.Laplacian(frame,  cv2.CV_64F)

    #hv = cv2.cvtColor(hv, cv2.COLOR_BGR2GRAY)
    hei, wid, o = frame.shape
    '''if cont == 200:
      mask = subtractor1.apply(hv)
      cont = 0
'''
    mask = subtractor1.apply(hv)
    for i in range(0, 1, 1):
        mask1 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    for i in range(0, 1, 1):
        mask2 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel)
    
    _, contours, _ = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        areaImage = wid*hei
        if area > (areaImage*0.001):
            cv2.drawContours(mask2, contours, i, (255,255,255), cv2.FILLED)
    _, contours, _ = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if cv2.contourArea(contour) < 700:
            continue
        cv2.rectangle(hv, (x,y), (x+w, y+h), (0,0,0), 3)
    print(hei, wid)
    output = np.ones((hei, wid*2), dtype="uint8")
    #v = cv2.hconcat([hv, mask])
    v = cv2.hconcat([hv, mask])
    output[0:hei, 0:wid*2] = v
    #v2 = cv2.hconcat([mask1, mask2])
    #output[hei:hei*2, 0:wid*2] = v2
    #v3 = cv2.vconcat([v, v2])
    '''
    perc = 50
    nw = int(v3.shape[1]*perc/100)
    nh = int(v3.shape[0]*perc/100)
    dim = (nw, nh)
    v_resized = cv2.resize(v3, dim, interpolation=cv2.INTER_AREA)
    # Write the frame into the file 'output.avi'
    print('frame size', frame.shape)
    print('v3 size', v3.shape)
    print('v3 resize', v_resized.shape)
    '''
    out.write(output)
    
    # Display the resulting frame    
    cv2.imshow('frame',v)
 
    # Press Q on keyboard to stop recording
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
 
  # Break the loop
  else:
    break

  cont += 1
  cv2.waitKey(20)
 
# When everything done, release the video capture and video write objects
cap.release()
out.release()
 
# Closes all the frames
cv2.destroyAllWindows() 