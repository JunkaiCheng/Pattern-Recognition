'''
Naive Bayes Classifiers on Digit classification
Single Pixel as feature

Training set: digitdata/optdigits-orig_train.txt
Testing set: digitdata/optdigits-orig_test.txt
Every 33 line one digit:
- 32 lines of binary pixels
- last line comtain the corresponding digit
'''

import numpy as np
import math
import time
import matplotlib.pyplot as plt

class NaiveBayes:
	def __init__(self, trainingData, testingData):
		self.trainingData = trainingData
		self.testingData = testingData
		self.guess = []
		self.matrix = []
		self.likelihood = {}
		self.occurence = {}
		self.totalNumOfTrainingTokens = {}
		self.prior = {}
		self.digitToListOfImage = {}

		for x in range(0,10):
			self.prior[x] = 0.0
			self.totalNumOfTrainingTokens[x] = 0
			self.digitToListOfImage[x] = []
			for val in range(0,2):
				for i in range(0,32):
					for j in range(0,32):
						self.occurence[((i,j), x, val)] = 0

	def train(self, k = 1):
		for data in self.trainingData:
			for i in range(0,32):
				for j in range(0,32):
					if data[1][32*i+j] == 1:
						self.occurence[((i,j), data[0], 1)] += 1
					elif data[1][32*i+j] == 0:
						self.occurence[((i,j), data[0], 0)] += 1
			self.totalNumOfTrainingTokens[data[0]] += 1

		# calculate likelihood with Laplace smoothing with constant k
		for x in range(0,10):
			# self.prior[x] = np.exp(math.log(self.totalNumOfTrainingTokens[x]) - math.log(len(self.trainingData)))
			self.prior[x] = float(self.totalNumOfTrainingTokens[x]) / float(len(self.trainingData))
			for val in range(0,2):
				for i in range(0,32):
					for j in range(0,32):
						# self.likelihood[((i,j),x,val)] = np.exp(math.log(self.occurence[((i,j), x, val)] + k) - math.log(self.totalNumOfTrainingTokens[x] + 2*k))
						self.likelihood[((i,j),x,val)] = float(self.occurence[((i,j), x, val)] + k) / float(self.totalNumOfTrainingTokens[x] + 2*k)

	def estimation(self):
		post = 0
		temp = -10000
		finalPost = -1
		postProbPerDigit = []
		for data in self.testingData:
			for x in range(0,10):
				for i in range(0,32):
					for j in range(0,32):
						if self.likelihood[((i,j),x,data[1][32*i+j])] != 0:
							# print self.likelihood[((i,j),x,data[1][32*i+j])]
							post += math.log(self.likelihood[((i,j),x,data[1][32*i+j])])
				post += math.log(self.prior[x])
				postProbPerDigit.append((x,post))
				post = 0
			for postTuple in postProbPerDigit:
				if postTuple[1] > temp:
					temp = postTuple[1]
					finalPost = postTuple[0]
			self.guess.append(finalPost)
			# self.digitToListOfImage[digit] = list of tuple - (post prob, image)
			self.digitToListOfImage[finalPost].append((temp,data[1]))
			temp = -10000
			finalPost = -1
			postProbPerDigit = []
		return self.guess

	# the digit I am least sure about
	def least(self, digit):
		temp = 10000
		ret = []
		for probImage in self.digitToListOfImage[digit]:
			if probImage[0] < temp:
				temp = probImage[0]
				ret = probImage[1]

		print(' '.join(map(str, ret)))

	def drawTrainedDigit(self, digit = 8):
		expect = []
		for i in range(0,32):
			row =[]
			for j in range(0,32):
				row.append(math.log(self.likelihood[((i,j),digit,1)]))
			expect.append(row)
		plt.imshow(np.array(expect).reshape(32,32), cmap='jet', interpolation='nearest')
		plt.colorbar()
		plt.show()

	def overallAccuracy(self, guess):
		solution = []
		failureCount = 0
		for data in self.testingData:
			solution.append(data[0])
		for x in range(0,len(solution)):
			if solution[x] != self.guess[x]:
				failureCount += 1

		# accuracyPercent = np.exp(math.log(len(solution)-failureCount) - math.log(len(solution)))
		accuracyPercent = float(len(solution)-failureCount) / float(len(solution))
		print ("failureCount", failureCount)
		print ("totalCount", len(solution))
		print ("Over all accuracy percentage", accuracyPercent)

	# return accuracy of given digit
	def digitAccuracy(self, digit):	
		solution = []
		total = 0
		correct = 0
		for data in self.testingData:
			solution.append(data[0])
			if data[0] == digit:
				total += 1		
		for x in range(0,len(solution)):
			if solution[x] == self.guess[x] and solution[x] == digit:
				correct += 1
		print ("correct", correct)
		print ("total", total)
		accuracyPercent = float(correct) / float(total)
		# accuracyPercent = np.exp(math.log(correct) - math.log(total))
		print ("Accuracy percentage for ", digit, accuracyPercent)	

	def confusionMatrix(self):
		solution = []
		guessCount = {}
		digitCountForSolution = {}
		for data in self.testingData:
			solution.append(data[0])
		for r in range(0,10):
			digitCountForSolution[r] = 0
			for x in range(0,10):
				guessCount[(r, x)] = 0
		for data in self.testingData:
			digitCountForSolution[data[0]] += 1
		for r in range(0,10):
			for x in range(0,len(solution)):
				if solution[x] == r:
					guessCount[(r, self.guess[x])] += 1

		for r in range(0,10):
			row = []
			for c in range(0,10):
				if guessCount[(r, c)] != 0 and digitCountForSolution[r] != 0:
					row.append(float(guessCount[(r, c)]) / float(digitCountForSolution[r]))
					# row.append(np.exp(math.log(guessCount[(r, c)]) - math.log(digitCountForSolution[r])))
				else:
					row.append(0)
			self.matrix.append(row)

		print self.matrix

	# print the image of given digit that have the highest posterior prob in the training set
	def revisitMAX(self, digit):
		maxGuess = []
		temp = -10000
		most = []
		post = 0
		for data in self.testingData:
			if data[0] == digit:
				for i in range(0,32):
					for j in range(0,32):
						if self.likelihood[((i,j),digit,data[1][32*i+j])] != 0:
							post += math.log(self.likelihood[((i,j),digit,data[1][32*i+j])])
				post += math.log(self.prior[digit])
				maxGuess.append((post, data[1]))
				post = 0
		for x in range(0,len(maxGuess)):
			if maxGuess[x][0] > temp:
				temp = maxGuess[x][0]
				most = maxGuess[x][1]

		print(' '.join(map(str, most)))

	# print the image of given digit that have the lowest posterior prob in the training set
	def revisitMIN(self, digit):
		minGuess = []
		temp = 10000
		least = []
		post = 0
		for data in self.testingData:
			if data[0] == digit:
				for i in range(0,32):
					for j in range(0,32):
						if self.likelihood[((i,j),digit,data[1][32*i+j])] != 0:
							post += math.log(self.likelihood[((i,j),digit,data[1][32*i+j])])
				post += math.log(self.prior[digit])
				minGuess.append((post, data[1]))
				post = 0
		for x in range(0,len(minGuess)):
			if minGuess[x][0] < temp:
				temp = minGuess[x][0]
				least = minGuess[x][1]

		print(' '.join(map(str, least)))

	# display odd ratio of the the given digit a and b
	def oddRatio(self, a, b):
		expect = []
		odds = 0
		for i in range(0,32):
			row =[]
			for j in range(0,32):
				# up = float(self.likelihood[((i,j),a,1)]) / float(1-self.likelihood[((i,j),a,1)])
				# down = float(self.likelihood[((i,j),b,1)]) / float(1-self.likelihood[((i,j),b,1)])
				# temp = float(up) / float(down)
				temp = float(self.likelihood[((i,j),a,1)]) / float(self.likelihood[((i,j),b,1)])
				odds = math.log(temp)
				row.append(odds)
			expect.append(row)
		plt.imshow(expect, cmap='jet', interpolation='nearest', vmin = -3, vmax = 4)
		plt.colorbar()
		plt.show()


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
		if lineCount % 33 != 0:
			line = line.strip('')
			line = list(map(int, line))
			temp.extend(line)
		else:
			line = line.strip(' ')
			digit = int(line)
			origData.append((digit, np.asarray(temp)))
			temp = []
		lineCount += 1
	return origData


if __name__ == "__main__":
	start = time.time()
 
 	trainingData = readImagesFromFile("digitdata/optdigits-orig_train.txt")
 	testingData = readImagesFromFile("digitdata/optdigits-orig_test.txt")
 	naive = NaiveBayes(trainingData, testingData)
 	naive.train()
 	for num in range(0,10):
	 	naive.revisitMAX(num)
	 	print "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&"
	 	naive.revisitMIN(num)
	 	print "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&"
 	# naive.drawTrainedDigit(num)
 	# guess = naive.estimation()
 	# naive.least(num)
 	# for x in range(0,10):
 	# 	naive.digitAccuracy(x)
 	# naive.overallAccuracy(guess)
 	# naive.confusionMatrix()

 	# naive.drawTrainedDigit(3)
 	# naive.oddRatio(9,3)

	end = time.time()
	print(end - start)