'''
Naive Bayes Classifiers on Face classification
Single Pixel as feature

Training image set: facedata/facedatatrain
Training solution set: facedata/facedatatrainlabels
Testing image set: facedata/facedatatest
Testing solution set: facedata/facedatatestlabels

One face image: 10 * 60
0 - none edge; # - edge
'''

import numpy as np
import math
import time
import matplotlib.pyplot as plt

class NaiveBayes:
	def __init__(self, trainingData, testingData, trainingSol, testingSol):
		self.trainingData = trainingData
		self.testingData = testingData
		self.trainingSol = trainingSol
		self.testingSol = testingSol
		self.numOfFace = 0
		self.prior = {}
		self.guess = []
		self.matrix = []
		self.likelihood = {}
		self.occurence = {}

		for x in range(0,len(trainingSol)):
			if trainingSol[x] == '1':
				self.numOfFace += 1
		for face in range(0,2):
			for x in range(0,2):
				for i in range(0,70):
					for j in range(0,60):
						self.occurence[((i,j), x, face)] = 0

	def train(self, k = 1):
		index = 0
		# occurence(coordinate, pixel value, face or not)
		for data in self.trainingData:
			for i in range(0,70):
				for j in range(0,60):
					if data[60*i+j] == '#':
						self.occurence[((i,j),1,int(self.trainingSol[index]))] += 1
					elif data[60*i+j] == ' ':
						self.occurence[((i,j),0,int(self.trainingSol[index]))] += 1
			index += 1

		# calculate likelihood with Laplace smoothing with constant k
		for face in range(0,2):
			if face == 0:
				# not a face
				self.prior[0] = float(len(self.trainingData) - self.numOfFace) / float(len(self.trainingData))
			elif face == 1:
				self.prior[1] = float(self.numOfFace) / float(len(self.trainingData))
			for x in range(0,2):
				for i in range(0,70):
					for j in range(0,60):
						if face == 0:
							# not a face
							self.likelihood[((i,j),x,0)] = float(self.occurence[((i,j),x,0)] + k) / float((len(self.trainingData) - self.numOfFace) + 2*k)
						elif face == 1:
							self.likelihood[((i,j),x,1)] = float(self.occurence[((i,j),x,1)] + k) / float(self.numOfFace + 2*k)

	def estimation(self):
		post = 0
		temp = -100000
		finalPost = -1
		postProbPerDigit = []
		for data in self.testingData:
			for face in range(0,2):
				for i in range(0,70):
					for j in range(0,60):
						if data[60*i+j] == ' ':
							val = 0
						elif data[60*i+j] == '#':
							val = 1
						if self.likelihood[((i,j),val,face)] != 0:
							post += math.log(self.likelihood[((i,j),val,face)])
				post += math.log(self.prior[face])
				postProbPerDigit.append((face,post))
				post = 0
			for postTuple in postProbPerDigit:
				if postTuple[1] > temp:
					temp = postTuple[1]
					finalPost = postTuple[0]
			self.guess.append(finalPost)
			temp = -100000
			finalPost = -1
			postProbPerDigit = []
		return self.guess

	def drawTrainedFace(self):
		expect = []
		for i in range(0,70):
			row =[]
			for j in range(0,60):
				row.append(self.likelihood[((i,j),1,1)])
			expect.append(row)
		plt.imshow(expect, cmap='jet', interpolation='nearest')
		plt.colorbar()
		plt.show()

	def overallAccuracy(self):
		failureCount = 0
		for x in range(0,len(self.testingSol)):
			if int(self.testingSol[x]) != self.guess[x]:
				failureCount += 1

		accuracyPercent = np.exp(math.log(len(self.testingSol)-failureCount) - math.log(len(self.testingSol)))
		print ("failureCount", failureCount)
		print ("totalCount", len(self.testingSol))
		print ("Over all accuracy percentage", accuracyPercent)

'''
Take in a file path, and parse the file
Return a list of tuples(digit, image)
'''
def readImagesFromFile(inputFilePath):
	file = open(inputFilePath, "r")
	lines = file.readlines()
	temp = []
	origData = []
	lineCount = 1
	for line in lines:
		line  = line.strip('\n')
		if lineCount % 70 != 0:
			line = line.strip('')
			line = list(map(str, line))
			temp.extend(line)
		else:
			line = line.strip('')
			line = list(map(str, line))
			temp.extend(line)
			origData.append(np.asarray(temp))
			temp = []
		lineCount += 1
	return origData


def readSolFromFile(inputFilePath):
	file = open(inputFilePath, "r")
	lines = file.readlines()
	origData = []
	for line in lines:
		line  = line.strip('\n')
		origData.append(line)
	return origData


if __name__ == "__main__":
	trainingData = readImagesFromFile("facedata/facedatatrain")
 	# print(' '.join(map(str, trainingData[0])))
 	testingData = readImagesFromFile("facedata/facedatatest")
 	trainingSol = readSolFromFile("facedata/facedatatrainlabels")
 	# print trainingSol
 	testingSol = readSolFromFile("facedata/facedatatestlabels")

 	naive = NaiveBayes(trainingData, testingData, trainingSol, testingSol)
 	naive.train()
 	# naive.drawTrainedFace()
 	naive.estimation()
 	naive.overallAccuracy()
