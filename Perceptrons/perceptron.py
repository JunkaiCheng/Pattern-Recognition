import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def readFile(input_filepath, bias = False):
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
            if bias:
                temp.append(1)
            orig_data.append((number, np.asarray(temp)))
            temp = []
        count += 1
    return orig_data

def init_zero_weights(length):
    return [float(0)]*length

def init_random_weights(length):
    result = []
    for i in range(length):
        result.append(random.random())
    return result

class Perceptron:
    def __init__(self, train_data, learning_rate, epochs, weight_vector, bias = False, weight_init = False):
        # bias: false = no bias, true = with bias
        # weight_init: false = zeros init, true = random init
        self.digit_class = np.asarray(list(range(10)))
        self.digit_class_weight = dict.fromkeys(self.digit_class)
        self.epochs = epochs
        self.train_data = train_data
        self.learning_rate = learning_rate
        for number, digit in enumerate(self.digit_class):
            self.digit_class_weight[digit] = np.asarray(weight_vector[number])

    def predict(self, x):
        calculated_digit = -1
        activation = -99999999999 # negative infinity
        for digit in self.digit_class:
            # feature_sum = 0
            feature_sum = np.inner(self.digit_class_weight[digit], x)
            if (feature_sum >= activation):
                activation = feature_sum
                calculated_digit = digit
        return calculated_digit

    def train(self):
        for epoch in range(self.epochs):
            self.learning_rate = 1/(epoch + 1)
            # self.learning_rate = 0.2/(epoch + 1)
            for data in self.train_data:
                calculated_digit = self.predict(data[1])
                expected_digit = data[0]
                if (calculated_digit != expected_digit):
                    self.digit_class_weight[expected_digit] += self.learning_rate * data[1]
                    self.digit_class_weight[calculated_digit] -= self.learning_rate * data[1]
            # self.learning_rate *= 0.99
        return self.digit_class_weight

    def evaluate(self, test_data):
        total_number = len(test_data)
        digit_sequence = []
        wrong = 0
        for data in test_data:
            expected_digit = data[0]
            feature_sum = -99999
            calculated_digit = -1
            for digit in self.digit_class:
                temp = np.inner(self.digit_class_weight[digit], data[1])
                if (temp > feature_sum):
                    feature_sum = temp
                    calculated_digit = digit
            if (calculated_digit != expected_digit):
                wrong += 1
            digit_sequence.append(calculated_digit)
        accuracy = 1 - wrong/total_number
        print ("The accuracy is:", accuracy)
        return accuracy, digit_sequence

if __name__ == "__main__":
    bias = True
    weight_init_random = True
    shuffle_train_set = True
    epochs = 26
    train_data = readFile("digitdata/optdigits-orig_train.txt", bias)
    test_data = readFile("digitdata/optdigits-orig_test.txt", bias)
    test_accuracy_list = []
    train_accuracy_list = []
    weight_vector_list = []

    if shuffle_train_set:
        random.shuffle(train_data)

    if weight_init_random:
        for i in range(10):
            weight_vector_list.append(init_random_weights(1024 + 1 * bias))
    else:
        for i in range(10):
            weight_vector_list.append(init_zero_weights(1024 + 1 * bias))

    expected_digit_sequence = []
    for data in test_data:
        expected_digit_sequence.append(data[0])

    for epoch in range(1, epochs, 1):
        print ("When epoch =", epoch)
        perceptron = Perceptron(train_data, 1, epoch, weight_vector_list, bias, weight_init_random)
        final_weight_vector = perceptron.train()
        test_accuracy, calculate_digit_sequence = perceptron.evaluate(test_data)
        test_accuracy_list.append(test_accuracy)
        train_accuracy_list.append(perceptron.evaluate(train_data)[0])

    CMatrix = confusion_matrix(expected_digit_sequence, calculate_digit_sequence, labels = range(10))
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
    plt.scatter(list(range(1, epochs, 1)), test_accuracy_list)
    plt.title("The accuracy of testing set")
    fig.savefig("test_accuracy.png")
    plt.close(fig)
    fig = plt.figure()
    plt.scatter(list(range(1, epochs, 1)), train_accuracy_list)
    plt.title("The accuracy of training set")
    fig.savefig("train_accuracy.png")
    plt.close(fig)

    weight_matrix_list = []
    for number in range(10):
        weight_matrix = []
        for i in range(32):
            temp = []
            for j in range(32):
                num = final_weight_vector[number][32 * i + j]
                temp.append(num)
            weight_matrix.append(temp)
        weight_matrix = np.matrix(weight_matrix)
        weight_matrix_list.append(weight_matrix)
    for number in range(10):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_aspect('equal')
        plt.imshow(weight_matrix_list[number], interpolation = 'nearest', cmap = 'jet')
        plt.colorbar()
        plt.title("The weight visualization figure for digit " + str(number))
        fig.savefig("weight" + str(number) + ".png")
        plt.close(fig)