import pandas as pd
import numpy
import scipy.special

# neural network class definition


class neuralNetwork:

    # initialize the neural network
    def __init__(self, inputnodes, hiddennodes, outputnods, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnods
        self.lr = learningrate

        # for the weightnet is unknown we just use random matrix
        self.wih = numpy.random.rand(self.hnodes, self.inodes) - 0.5
        self.who = numpy.random.rand(self.onodes, self.hnodes) - 0.5

    # train the neural network

    def train(self, input_list, target_list):
        # convert the input_list and target_list to 2d-array
        inputs = numpy.array(input_list, ndmin=2).T
        targets = numpy.array(target_list, ndmin=2).T
        # we use Scipy to find function to cater the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)

        # connection between the statics and the weight matrixs
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        # error is the final_output - the targets
        outputs_error = targets - final_outputs

        # error is the final_output - the targets
        hidden_error = numpy.dot(self.who.T, outputs_error)

        # update the weight matrix between hidden and output layer
        self.who += self.lr * numpy.dot((outputs_error * final_outputs * (
            1.0 - final_outputs)), numpy.transpose(hidden_outputs))

        # update the weight matrix between hidden and input layer
        self.wih += self.lr * numpy.dot((hidden_error * hidden_outputs * (
            1.0 - hidden_outputs)), numpy.transpose(hidden_outputs))

    # query the neural network

    def query(self, input_list):
        # convert the input_list to 2d-array
        inputs = numpy.array(input_list, ndmin=2).T

        # we use Scipy to find function to cater the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)

        # connection between the statics and the weight matrixs
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


test01 = 'BTC-USD10.csv'
test01_target = pd.read_csv(test01)['Close']
test01_target = test01_target.values.tolist()

test01_inputs = list(range(1, 433))
# print(test01_target)

# number of input, hidden and output nodes
input_nodes = 432
hidden_nodes = 432
output_nodes = 2

# learning rate is 0.3
learning_rate = 0.3

# create instance of neural network
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# n.train(test01_inputs, test01_target)

# test01_pre_output = list(range(2, 434))

# print(n.query(test01_pre_output))
