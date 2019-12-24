import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt


class Filtering () :

    def filter (image, output):
        output = cv.medianBlur(image, (11,11))
        return output

    def fillContours(image, output, value, cols, rows):
        contours, hierarchy = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for i in range(len(contours)):
            area = cv.contourArea(contour)
            areaImage = cols*rows
            if area > (areaImage*0.001):
                cv.drawContours(output, contours, i, np.scalar(value), cv.CV_FILLED)
    
    def fillConvexHull(image, output, value) :
        contours, hierarchy = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        hull = []
        for i in range(len(contours)):
            hull.append(convexHull(contours[i], False))
        for i in range(len(hull)):
            area = cv.contourArea(hull[i])
            areaImage = cols*rows
            if area > (areaImage*0.001):
                cv.drawContours(output, hull, i, np.scalar(value), cv.CV_FILLED)
