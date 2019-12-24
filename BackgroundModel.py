import cv2 as cv
import numpy as np
from PixelGMMZ import PixelGMMZ

class PixelBackgroundModel():
	def __init__ (self, fAlphaT, fTb, fTg, fTB, fSigma, fMinSigma, fMaxSigma, fCT, fDnorm,
		nM, bShadowDetection, fTau, nNBands, nWidth, nHeight, nSize, line):
		self.fAlphaT = fAlphaT
		self.fTb = fTb#threshold mahala
		self.fTg = fTg  #check fit
		self.fTB = fTB #threshold weight
		self.fSigma = fSigma #default sigma
		self.fMinSigma = fMinSigma
		self.fMaxSigma = fMaxSigma
		self.fCT = fCT #max foreground data
		self.fDnorm = fDnorm #norm distance (default=7.0)
		self.nM = nM
		self.bShadowDetection = bShadowDetection
		self.fTau = fTau
		self.nNBands = nNBands
		self.nWidth = nWidth
		self.nHeight = nHeight
		self.nSize = nSize
		self.line = line
		self.roi_x = 0
		self.roi_y = 0
		self.roi_width = nWidth
		self.roi_height = nHeight
		self.pixelGMMZ = PixelGMMZ()
    
	def _backgroundModel(posPixel, red, green, blue, distance, dnorm, pModeUsed, m_aGaussians, m_nM, m_fAlphaT, m_fTb, m_fTB, m_fSigma, m_fMaxSifma, m_fPrune):
		pos = 0
		bFitsPDF = False
		bBackground = False
		m_fOneMinAlpha = 1 - m_fAlphaT
		nModes = pModesUsed
		totalWeight = 0.0
		dstFinal = dnorm
		#std::cout << "entrou 4" << std::endl
		for iModes in nModes:
			pos = posPixel + iModes
			weight = m_aGaussians[pos].weight

			if not bFitsPDF:
				var = m_aGaussians[pos].sigma
				muR = m_aGaussians[pos].muR
				muG = m_aGaussians[pos].muG
				muB = m_aGaussians[pos].muB

				dR = muR - red
				dG = muG - green
				dB = muB - blue

				dist = (dR * dR + dG * dG + dB * dB)
				mahala = dist / var
				#background? - m_fTb
				if (totalWeight < m_fTB) and (dist < m_fTb * var):
					bBackground = 1
				if (totalWeight < m_fTB) and (mahala < dstFinal):
					dstFinal = mahala
				#check fit
				if dist < (m_fTg * var):
					#belongs to the mode
					bFitsPDF = 1

					#update distribution
					k = m_fAlphaT / weight
					weight = m_fOneMinAlpha * weight + m_fPrune
					weight += m_fAlphaT
					m_aGaussians[pos].muR = muR - k * (dR)
					m_aGaussians[pos].muG = muG - k * (dG)
					m_aGaussians[pos].muB = muB - k * (dB)

					sigmanew = var + k * (dist - var)

					if sigmanew < m_fMinSigma:
						m_aGaussians[pos].sigma = m_fMinSigma
					else:
						if sigmanew > m_fMaxSigma:
							m_aGaussians[pos].sigma = m_fMaxSigma
						else:
							m_aGaussians[pos].sigma = sigmanew
					
					for i in range(iMode, 0, -1) :
						posLocal = posPixel + iLocal
						if weight < (m_aGaussians[posLocal - 1].weight):
							break
						else:
							temp = m_aGaussians[posLocal]
							m_aGaussians[posLocal] = m_aGaussians[posLocal - 1]
							m_aGaussians[posLocal - 1] = temp

				else:
					weight = m_fOneMinAlpha * weight + m_fPrune
					if weight < -m_fPrune:
						weight = 0.0
						nModes -= 1
						#bPrune= 1
						#break;//the components are sorted so we can skip the rest
				#check if it fits the current mode (2.5 sigma)
			#fit not found yet
			else:
				weight = m_fOneMinAlpha * weight + m_fPrune
				#check prune
				if weight < -m_fPrune:
					weight = 0.0
					nModes -= 1
					#bPrune=1
					#break;//the components are sorted so we can skip the rest
			totalWeight += weight
			m_aGaussians[pos].weight = weight
		
		#////std::cout << "saiu 4" << std::endl;
		#//go through all modes
		#//////

		#//renormalize weights

		#////std::cout << "entrou 5" << std::endl;
		for iLocal in range(nModes):
			m_aGaussians[posPixel + iLocal].weight = m_aGaussians[posPixel + iLocal].weight / totalWeight

		#//make new mode if needed and exit
		if not bFitsPDF:
			if nModes == m_nM:
				pass
			else:
				#//add a new one
				#//totalWeight+=m_fAlphaT;
				#//pos++;
				nModes += 1
			pos = posPixel + nModes - 1
			if (nModes == 1):
				m_aGaussians[pos].weight = 1
			else:
				m_aGaussians[pos].weight = m_fAlphaT

			#//renormalize weights
			for iLocal in range(nModes - 1):
				m_aGaussians[posPixel + iLocal].weight *= m_fOneMinAlpha

			m_aGaussians[pos].muR = red
			m_aGaussians[pos].muG = green
			m_aGaussians[pos].muB = blue
			m_aGaussians[pos].sigma = m_fSigma

			#//sort
			#//find the new place for it
			for iLocal in range((nModes - 1), 0, -1):
				posLocal = posPixel + iLocal
				if m_fAlphaT < m_aGaussians[posLocal - 1].weight:
					break
				else:
					#//swap
					temp = m_aGaussians[posLocal]
					m_aGaussians[posLocal] = m_aGaussians[posLocal - 1]
					m_aGaussians[posLocal - 1] = temp

		#////std::cout << "saiu 5" << std::endl;

		#//set the number of modes

		pModesUsed = nModes

		#////std::cout << "entrou 6" << std::endl;
		distance = (dstFinal / dnorm)
		#////std::cout << "saiu 6" << std::endl;
		return bBackground

	def _removeShadow (posPixel, red, green, blue, chroma, lumina, dnorm, nModes, m_aGaussians,m_nM, m_fTb, m_fTB, m_fTg, m_fTau):
		pos = 0
		tWeight = 0.0
		numerador, denominador = 0.0

		for modes in nModes : 
			pos = posPixel + modes
			var = m_aGaussians[pos].sigma
			muR = m_aGaussians[pos].muR
			muG = m_aGaussians[pos].muG
			muB = m_aGaussians[pos].muB
			weight = m_aGaussians[pos].weight
			tWeight += weight

			numerator = (red * muR) + (green * muG) + (blue * muB)
			denominator = (muR * muR) + (muG * muG) + (muB * muB)
			if denominator == 0:
				break
			
			a = numerator / denominator
			lumina = a

			if (a <= 1) and (a >= m_fTau):
				dR = a * muR - red
				dG = a * muG - green
				dB = a * muB - blue
				dist = (dR * dR) + (dG *dG) + (dB* dB)
				chroma = (dist / (var * a * a)) / dnorm

				if dist < (m_fTb * var * a * a):
					return 2
			
			if tWeight > m_fTB:
				break
		
		return 0

	def setRoi (self, x, y, width, height):
		self.roi_x = 0
		self.roi_y = 0
		self.roi_width = width
		self.roi_height = height

