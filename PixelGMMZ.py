import cv2
import numpy as np 

class PixelGMMZ ():

    def __init__(self, sigma = 0, muR = 0, muG = 0, muB = 0, weight = 0):
        self.sigma = sigma
        self.muR = muR
        self.muG = muG
        self.muB = muB
        self.weight = weight
    
    def setSigma(self, sigma):
        self.sigma = sigma
    
    def setMuR(self, muR):
        self.muR = muR
    
    def setMuG(self, muG):
        self.muG = muG
    
    def setMuB(self, muB):
        self.muB = muB
    
    def setWeight(self, weight):
        self.weight = weight