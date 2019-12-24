import matplotlib.pyplot as plt 
import numpy as np 
import cv2 as cv
import sys

class Classifier ():
    THRESHOLD_METHOD_KITTLER = 1
    THRESHOLD_METHOD_KAPUR = 2
    THRESHOLD_METHOD_YEN = 4
    THRESHOLD_METHOD_OTSU = 8
    THRESHOLD_METHOD_PAL_1 = 16
    THRESHOLD_METHOD_PAL_2 = 32
    THRESHOLD_TYPE_BINARY = 64
    THRESHOLD_TYPE_BINARY_INV = 128

    def getOtsuThreshold(image, cols, rows):
        threshold = 0
        thresh = sys.float_info.max
        iSize = cols*rows
        #intializing histogram
        hist = []
        for i in range(256):
            hist.append(0)
        
        #calcu histogram
        for i in range(iSize):
            hist[image[i]] = hist[image[i] + 1]
        
        #normalizing
        prob = [(hist[0]*(1./iSize))]
        acc = [prob[0]]
        
        for i in range(256):
            prob.append(hist[i]*(1./iSize))
            acc.append(prob[i]+acc[i-1])
        
        #calculating threshold
        sum_gl = 0
        sum_lG = 0
        mu0, mu1, std0, std1 = []
        for i in range(255):
            sum_gl = 0
            sum_lG = 0
            for j in range (i):
                sum_gl += ((j-1)*prob[j]) 
            for j in range(256):
                j += 1
                sum_lG += ((j+1)*prob[j])
            mu0[i] = sum_gl/acc[i]
            mu1[i] = sum_lG/(1-acc[i])
        
        for i in range(255):
            sum_gl = 0
            sum_lG = 0
            err = 0.0
            for j in range(i):
                err = (j+1) - mu0[i]
                err = err*err
                sum_gl += err*prob[j]
            
            for j in range (256):
                err = (j+1) - mu1[i]
                err = err*err
                sum_lG += err*prob[j]
            std0[i] = (sum_gl/acc[i])**(1/2)
            std1[i] = (sum_lG/(1-acc[i]))**(1/2)
        
        for i in range(256):
            th = 1
            if std0[i] != 0:
                th = 2*acc[i]*math.log(std0[i])
            if std1[i] != 0:
                th = 2*(1-acc[i])*math.log(std1[i])
            if acc[i] != 0:
                th = -2*acc[i]*math.log(acc[i])
            if (1-acc[i]) != 0:
                th = -2*(1-acc[i])*math.log(1-acc[i])
            
            if th < thresh:
                thresh = th
                threshold = 1
            
        return threshold
    
    def getKittlerThreshold(image, cols, rows):
        threshold = 0
        thresh = sys.float_info.max
        iSize = cols*rows
        #intializing histogram
        hist = []
        for i in range(256):
            hist.append(0)
        
        #calcu histogram
        for i in range(iSize):
            hist[image[i]] = hist[image[i] + 1]
        
        #normalizing
        prob = [(hist[0]*(1./iSize))]
        acc = [prob[0]]
        
        for i in range(256):
            prob.append(hist[i]*(1./iSize))
            acc.append(prob[i]+acc[i-1])
        
        #calculating threshold
        sum_gl = 0
        sum_lG = 0
        mu0, mu1, std0, std1 = []
        for i in range(255):
            sum_gl = 0
            sum_lG = 0
            for j in range (i):
                sum_gl += ((j-1)*prob[j]) 
            for j in range(256):
                j += 1
                sum_lG += ((j+1)*prob[j])
            mu0[i] = sum_gl/acc[i]
            mu1[i] = sum_lG/(1-acc[i])
        
        for i in range(255):
            sum_gl = 0
            sum_lG = 0
            err = 0.0
            for j in range(i):
                err = (j+1) - mu0[i]
                err = err*err
                sum_gl += err*prob[j]
            
            for j in range (256):
                err = (j+1) - mu1[i]
                err = err*err
                sum_lG += err*prob[j]
            std0[i] = (sum_gl/acc[i])**(1/2)
            std1[i] = (sum_lG/(1-acc[i]))**(1/2)
        
        for i in range(256):
            th = 1
            aux = 0
            if std0[i] != 0:
                aux = 2*acc[i]*math.log(std0[i])
                th += aux
            if std1[i] != 0:
                aux = 2*(1-acc[i])*math.log(std1[i])
                th += aux
            if acc[i] != 0:
                aux = -2*acc[i]*math.log(acc[i])
                th += aux
            if (1-acc[i]) != 0:
                aux = -2*(1-acc[i])*math.log(1-acc[i])
                th += aux
            
            if th < thresh:
                thresh = th
                threshold = 1
            
        return threshold

    def getYenThreshould(image, cols, rows ):
        threshold = 0
        iSize = cols*rows
        best_entropy = 0.0

        #intializing histogram
        hist = []
        for i in range(256):
            hist.append(0)
        
        #calcu histogram
        for i in range(iSize):
            hist[image[i]] = hist[image[i] + 1]
        
        #normalizing
        prob = [(hist[0]*(1./iSize))]
        acc = [prob[0]]
        
        for i in range(256):
            prob.append(hist[i]*(1./iSize))
            acc.append(prob[i]+acc[i-1])
        #calculating threshold
        for i in range(255): 
            sum_gl = 0
            sum_lG = 0
            gpt, gpot, entropy

            for j in range(i):
                gpt = prob[j]/acc[i]
                sum_gl += (gpt*gpt)
            
            for j in range(i):
                gpot = prob[j]/(1-acc[i])
                sum_lG += (gpot*gpot)
            if (sum_gl > 1e-5) and (sum_lG > 1e-5):
                entropy = -math.log2(sum_gl) - math.log2(sum_lG)
            if entropy > best_entropy:
                best_entropy = entropy
                threshold = 1
        
        return threshold
    
    def getKapurThreshold(image, cols, rows):
        threshold = 0
        iSize = cols*rows
        best_entropy = 0.0

        #intializing histogram
        hist = []
        for i in range(256):
            hist.append(0)
        
        #calcu histogram
        for i in range(iSize):
            hist[image[i]] = hist[image[i] + 1]
        
        #normalizing
        prob = [(hist[0]*(1./iSize))]
        acc = [prob[0]]
        
        for i in range(256):
            prob.append(hist[i]*(1./iSize))
            acc.append(prob[i]+acc[i-1])
        
        #calculating threshold
        for i in range(255): 
            sum_gl = 0
            sum_lG = 0
            gpt, gpot, entropy

            for j in range(i):
                gpt = prob[j]/acc[i]
                if gpt > 1e-5 :
                    sum_gl += gpt*math.log2(gpt)
            
            for j in range(i):
                gpot = prob[j]/(1-acc[i])
                if gpot > 1e-5:
                    sum_lG += gpot*math.log2(gpot)
            entropy = -sum_gl - sum_lG
            if entropy > best_entropy:
                best_entropy = entropy
                threshold = 1
    
        return threshold
    
    def getSpartialPal1Threshold(image, cols, rows):
        histNorm = 0
        thr = 0
        best_entropy = 0.0
        hist = []
        for i in range (256):
            for j in range (256):
                hist[i*256+j] = 0
        
        for i in range(rows):
            for j in range(cols):
                hist[image[i*cols+j]*256+image[i*cols(j+1)]] = hist[image[i*cols+j]*256+image[i*cols(j+1)]] + 1
                hist[image[i*cols+j]*256+image[(i+1)*cols+j]] = hist[image[i*cols+j]*256+image[(i+1)*cols+j]] + 1
                histNorm = histNorm + 2
        prob = []
        acc = []
        for i in range(256):
            for j in range(256):
                prob[i*256+j] = hist[i*256+j]/histNorm
                acc[i*256+j] = prob[i*256+j]
                if i > 0:
                    acc[i*256+j] = prob[i*256+j]+acc[(i-1)*256+j]
                if j > 0:
                    acc[i*256+j] = prob[i*256+j]+acc[i*256+(j-1)]
                if i > 0 and j > 0:
                    acc[i*256+j] = prob[i*256+j]+acc[(i-1)*256+(j-1)]
        
        for i in range(255):
            Pa = acc[i*256+i]
            Pb = acc[i*256+255]-acc[i*256+i]
            Pd = acc[255*256+i]-acc[i*256+i]
            Pc = acc[255*256+255]-Pa-Pb-Pd
            gpt, gpot, tnropy, val = 0.0

            for j in range(i):
                for k in range(i):
                    val = prob[i*256+j]/Pa
                    if val > 1e-5:
                        gpt += val*math.log(val)
            
            for j in range(256):
                for k in range(256):
                    val = prob[i*256+j]/Pc
                    if val > 1e-5:
                        gpt += val*math.log(val)
            
            entropy = -gpt/2.0 - gpot/2.0
            if entropy > best_entropy :
                best_entropy = entropy
                thr = 1
        
        return thr
    
    def getSpartialPal2Threshold(image, cols, rows):
        histNorm = 0
        thr = 0
        best_entropy = 0.0
        hist = []
        for i in range (256):
            for j in range (256):
                hist[i*256+j] = 0
        
        for i in range(rows):
            for j in range(cols):
                hist[image[i*cols+j]*256+image[i*cols(j+1)]] = hist[image[i*cols+j]*256+image[i*cols(j+1)]] + 1
                hist[image[i*cols+j]*256+image[(i+1)*cols+j]] = hist[image[i*cols+j]*256+image[(i+1)*cols+j]] + 1
                histNorm = histNorm + 2
        prob = []
        acc = []
        for i in range(256):
            for j in range(256):
                prob[i*256+j] = hist[i*256+j]/histNorm
                acc[i*256+j] = prob[i*256+j]
                if i > 0:
                    acc[i*256+j] = prob[i*256+j]+acc[(i-1)*256+j]
                if j > 0:
                    acc[i*256+j] = prob[i*256+j]+acc[i*256+(j-1)]
                if i > 0 and j > 0:
                    acc[i*256+j] = prob[i*256+j]+acc[(i-1)*256+(j-1)]
        
        for i in range(255):
            #Pa = acc[i*256+i]
            Pb = acc[i*256+255]-acc[i*256+i]
            Pd = acc[255*256+i]-acc[i*256+i]
            #Pc = acc[255*256+255]-Pa-Pb-Pd
            gpt, gpot, tnropy, val = 0.0

            for j in range(i):
                for k in range(i):
                    val = prob[i*256+j]/Pd
                    if val > 1e-5:
                        gpt += val*math.log(val)
            
            for j in range(256):
                for k in range(256):
                    val = prob[i*256+j]/Pb
                    if val > 1e-5:
                        gpt += val*math.log(val)
            
            entropy = -gpt/2.0 - gpot/2.0
            if entropy > best_entropy :
                best_entropy = entropy
                thr = 1
        
        return thr
    
    def threshold(self, image, output, flag, cols, rows):
        thresh = 0
        if (THRESHOLD_METHOD_KITTLER & flag) != 0 :
            thresh = self.getKittlerThreshold(image, cols, rows)
        elif (THRESHOLD_METHOD_KAPUR & flag) != 0 :
            thresh = self.getKapurThreshold(image, cols, rows)
        elif (THRESHOLD_METHOD_OTSU & flag) != 0 :
            thresh = self.getOtsuThreshold(image, cols, rows)
        elif (THRESHOLD_METHOD_PAL_1 & flag) != 0 :
            thresh = self.getSpartialPal1Threshold(image, cols, rows)
        elif (THRESHOLD_METHOD_PAL_2 & flag) != 0 :
            thresh = self.getSpartialPal2Threshold(image, cols, rows)
        
        if (THRESHOLD_TYPE_BINARY & flag) != 0 :
            output = cv.threshold(image, thresh, 255, cv.THRESH_BINARY)
        elif (THRESHOLD_TYPE_BINARY_INV & flag) != 0 :
            output = cv.threshold(image, thresh, 255, cv.THRESH_BINARY_INV)
    
    def thresholdMatrix(image, thresh, output, flag, cols, rows):
        for i in range(rows):
            for j in range(cols):
                if (THRESHOLD_TYPE_BINARY & flag) != 0 :
                    if image[i*cols+j] > thresh[i*cols+j]:
                        output[i*cols+j] = 255
                    else :
                        output[i*cols+j] = 0
                elif (THRESHOLD_TYPE_BINARY_INV & flag) != 0 :
                    if image[i*cols+j] > thresh[i*cols+j]:
                        output[i*cols+j] = 0
                    else :
                        output[i*cols+j] = 255
    
    