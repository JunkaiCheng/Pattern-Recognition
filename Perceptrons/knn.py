import numpy as np
import time
import math
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def readFile(input_filepath):
    file = open(input_filepath, "r")
    lines = file.readlines()
    temp = []
    orig_data = []
    count = 1
    for line in lines:
        line = line.strip('\n')
        if count % 33 != 0:
            line = line.strip('')
            line = list(map(int, line))
            temp.extend(line)
        else:
            line = line.strip(' ')
            number = int(line)
            orig_data.append((number, np.asarray(temp)))
            temp = []
        count += 1
    return orig_data

def euclideanDistance(vector1, vector2):
    # summation = np.sum(abs(vector1 - vector2))
    summation = np.sum(np.power((vector1 - vector2), 2))
    return math.sqrt(summation)

def getNeighbor(test, trainingSet, k):
    originalData = []
    for instance in trainingSet:
        distance = euclideanDistance(test[1], instance[1])
        originalData.append((instance[0], distance))
    originalData = sorted(originalData, key = lambda x: -x[0])
    originalData = sorted(originalData, key = lambda x: x[1])
    neighbors = []
    for i in range(k):
        neighbors.append(originalData[i])
    return neighbors

def predict(test, trainingSet, k):
    neighbors = getNeighbor(test, trainingSet, k)
    digit_list = []
    for neighbor in neighbors:
        digit_list.append(neighbor[0])
    digit_dict = defaultdict(int)
    for digit in digit_list:
        digit_dict[digit] += 1
    calculated_digit = -1
    occurence = -1
    for digit in digit_list:
        if digit_dict[digit] > occurence:
            calculated_digit = digit
            occurence = digit_dict[digit]
    return calculated_digit

def getAccuracy(testSet, trainingSet, k):
    wrong = 0
    expected_digit_sequence = []
    calculated_digit_sequence = []
    for test in testSet:
        expected_digit_sequence.append(test[0])
        calculated_digit = predict(test, trainingSet, k)
        calculated_digit_sequence.append(calculated_digit)
        if test[0] != calculated_digit:
            wrong += 1
    accuracy = 1 - wrong / len(testSet)
    print ("The accuracy when k =", k, "is", accuracy)
    return accuracy, expected_digit_sequence, calculated_digit_sequence

if __name__ == "__main__":
    train_data = readFile("digitdata/optdigits-orig_train.txt")
    test_data = readFile("digitdata/optdigits-orig_test.txt")
    # start_time = time.time()
    # predict(test_data[0], train_data, 1)
    accuracyList = []
    for k in range(1, 26, 1):
        accuracy = getAccuracy(test_data, train_data, k)[0]
        accuracyList.append(accuracy)
    # print("--- %s seconds ---" % (time.time() - start_time))

    accuracy, expected_digit_sequence, calculated_digit_sequence = getAccuracy(test_data, train_data, 1)
    CMatrix = confusion_matrix(expected_digit_sequence, calculated_digit_sequence, labels = range(10))
    summation = np.sum(CMatrix, axis = 1)
    finalCMatrix = []
    for i in range(10):
        temp = []
        for j in range(10):
            temp.append(CMatrix[i][j] * 100 / summation[i])
        finalCMatrix.append(temp)
    print ("The Confusion Matrix is as follows:")
    print (finalCMatrix)

    fig = plt.figure()
    plt.scatter(list(range(1, 26, 1)), accuracyList)
    plt.ylim(0.975, 1)
    plt.title("The accuracy of testing set")
    fig.savefig("test_accuracy for knn.png")
    plt.close(fig)