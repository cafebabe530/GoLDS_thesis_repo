#Dedicated Artificial Neural Network for GoLDS Gesture Recognition feature.
#2-layer Neural Network
#Number of neurons per layer: 12-10-10-8

import numpy as np
import random
import time

#Sigmoid function
def sigmoid(i):
    return 1/(1 + np.exp(-i))

#Derivative of Sigmoid function
def derivatives_sigmoid(i):
    return i * (1 - i)

#Neural Network class
class NeuNet:
    
    def __init__(self):
        #initialization
        self.IL_neurons = 12 #input layer neurons = 12
        self.HL_neurons = 10 #hidden layer neurons = 10
        self.OL_neurons = 8 #output layer neurons = 8

        #weights of input-to-hidden1 layer, 12x10
        self.wh1 = np.random.uniform(size=(self.IL_neurons, self.HL_neurons))
        #bias of input-to-hidden1 layer, 1x10
        self.bh1 = np.random.uniform(size=(1, self.HL_neurons))
        #weights of hidden1-to-hidden2 layer, 10x10
        self.wh2 = np.random.uniform(size=(self.HL_neurons, self.HL_neurons))
        #bias of hidden1-to-hidden2 layer, 1x10
        self.bh2 = np.random.uniform(size=(1, self.HL_neurons))
        #weights of hidden2-to-output layer, 10x8
        self.wout = np.random.uniform(size=(self.HL_neurons, self.OL_neurons))
        #bias of hidden2-to-output layer, 1x8
        self.bout = np.random.uniform(size=(1, self.OL_neurons))

        #this is just to declare variables for loaded weights and biases
        
        self.loadwh1 = np.random.uniform(size=(self.IL_neurons, self.HL_neurons))
        self.loadbh1 = np.random.uniform(size=(1, self.HL_neurons))
        
        self.loadwh2 = np.random.uniform(size=(self.HL_neurons, self.HL_neurons))
        self.loadbh2 = np.random.uniform(size=(1, self.HL_neurons))
        
        self.loadwout = np.random.uniform(size=(self.HL_neurons, self.OL_neurons))
        self.loadbout = np.random.uniform(size=(1, self.OL_neurons))

    def train(self, inPattern, outPattern):
        #function to train system
        
        #store input pattern to local list
        self.inArray = np.array(inPattern)
        #store output pattern to local list
        self.outArray = np.array(outPattern)

        self.epochs = 17000 #training iterations
        self.lr = 0.1 #learning rate

        for self.j in range(self.epochs):

        #forward propagation
            #hidden layer1
            
            self.weighted_sum_HL1 = np.dot(self.inArray, self.wh1) #weighted sum no bias
            self.weighted_sum_HL_withBias1 = self.weighted_sum_HL1+ self.bh1 #weighted sum + bias
            self.hiddenlayer1 = sigmoid(self.weighted_sum_HL_withBias1) #activation function

            #hidden layer2
            self.weighted_sum_HL2 = np.dot(self.hiddenlayer1, self.wh2) #weighted sum no bias
            self.weighted_sum_HL_withBias2 = self.weighted_sum_HL2 + self.bh2 #weighted sum + bias
            self.hiddenlayer2 = sigmoid(self.weighted_sum_HL_withBias2) #activation function

            #output layer
            self.weighted_sum_OL = np.dot(self.hiddenlayer2, self.wout) #weighted sum no bias
            self.weighted_sum_OL_withBias = self.weighted_sum_OL + self.bout #weighted sum + bias
            self.output = sigmoid(self.weighted_sum_OL_withBias) #activation function

        #backpropagation
            #get errors
            self.error = self.outArray - self.output
            self.slope_output_layer = derivatives_sigmoid(self.output)
            self.slope_hidden_layer2 = derivatives_sigmoid(self.hiddenlayer2)
            self.slope_hidden_layer1 = derivatives_sigmoid(self.hiddenlayer1)
            #output layer
            self.d_output = self.error * self.slope_output_layer
            #hidden layer 2
            self.error_at_hidden_layer2 = self.d_output.dot(self.wout.T)
            self.d_hiddenlayer2 = self.error_at_hidden_layer2 * self.slope_hidden_layer2
            #hidden layer 1
            self.error_at_hidden_layer1 = self.d_hiddenlayer2.dot(self.wh2.T)
            self.d_hiddenlayer1 = self.error_at_hidden_layer1 * self.slope_hidden_layer1
            
            #update weights and bias
            #output
            self.wout += self.hiddenlayer2.T.dot(self.d_output) *self.lr
            self.bout += np.sum(self.d_output, axis=0, keepdims=True) *self.lr
            #hidden 2
            self.wh2 += self.hiddenlayer1.T.dot(self.d_hiddenlayer2) *self.lr
            self.bh2 += np.sum(self.d_hiddenlayer2, axis=0, keepdims=True) *self.lr
            #hidden 1
            self.wh1 += self.inArray.T.dot(self.d_hiddenlayer1) *self.lr
            self.bh1 += np.sum(self.d_hiddenlayer1, axis=0, keepdims=True) *self.lr
        
        print("Training is complete!")
        #save weights and biases
        saveFile = open('weightsHidden1.txt', 'w')
        saveFile.write(str(self.wh1))
        saveFile.close()

        saveFile = open('biasHidden1.txt', 'w')
        saveFile.write(str(self.bh1))
        saveFile.close()

        saveFile = open('weightsHidden2.txt', 'w')
        saveFile.write(str(self.wh2))
        saveFile.close()

        saveFile = open('biasHidden2.txt', 'w')
        saveFile.write(str(self.bh2))
        saveFile.close()

        saveFile = open('weightsOutput.txt', 'w')
        saveFile.write(str(self.wout))
        saveFile.close()

        saveFile = open('biasOutput.txt', 'w')
        saveFile.write(str(self.bout))
        saveFile.close()

    def loadWB(self):
        #load weights and biases

        #HIDDEN LAYER 1 WEIGHTS
        #read, string file
        readFile = open('weightsHidden1.txt', 'r').read()
        #process string
        readFile = readFile.replace('[','', 150) #remove [
        readFile = readFile.replace(']','', 150) #remove ]
        readFile = readFile.replace('  ',' ', 150) #remove double white space
        readFile = readFile.replace('\n','', 150) #remove new line
        #put in list
        data = [str(x) for x in readFile.split(' ')] #list of strings
        #remove stray elements
        for x in data[:]:
            if(x == ''):
                data.remove(x)
        #convert each element to float
        for s in range(120):
            data[s] = float(data[s])        
        self.loadwh1 = [] # LOADED WEIGHTS H1
        count = 0
        for k in range(12):
            column = []
            for i in range(10):
                column.append(data[count])
                count += 1
            self.loadwh1.append(column)

        #HIDDEN LAYER 1 BIASES
        #read, string file
        readFile = open('biasHidden1.txt', 'r').read()
        #process string
        readFile = readFile.replace('[','', 150) #remove [
        readFile = readFile.replace(']','', 150) #remove ]
        readFile = readFile.replace('  ',' ', 150) #remove double white space
        readFile = readFile.replace('\n','', 150) #remove new line
        #put in list
        data = [str(x) for x in readFile.split(' ')] #list of strings
        #remove stray elements
        for x in data[:]:
            if(x == ''):
                data.remove(x)
        #convert each element to float
        for s in range(10):
            data[s] = float(data[s])
        self.loadbh1 = [] # LOADED BIASES H1
        count = 0
        for k in range(1):
            column = []
            for i in range(10):
                column.append(data[count])
                count += 1
            self.loadbh1.append(column)

        #HIDDEN LAYER 2 WEIGHTS
        #read, string file
        readFile = open('weightsHidden2.txt', 'r').read()
        #process string
        readFile = readFile.replace('[','', 150) #remove [
        readFile = readFile.replace(']','', 150) #remove ]
        readFile = readFile.replace('  ',' ', 150) #remove double white space
        readFile = readFile.replace('\n','', 150) #remove new line
        #put in list
        data = [str(x) for x in readFile.split(' ')] #list of strings
        #remove stray elements
        for x in data[:]:
            if(x == ''):
                data.remove(x)
        #convert each element to float
        for s in range(100):
            data[s] = float(data[s])
        self.loadwh2 = [] # LOADED WEIGHTS H2
        count = 0
        for k in range(10):
            column = []
            for i in range(10):
                column.append(data[count])
                count += 1
            self.loadwh2.append(column)

        #HIDDEN LAYER 2 BIASES
        #read, string file
        readFile = open('biasHidden2.txt', 'r').read()
        #process string
        readFile = readFile.replace('[','', 150) #remove [
        readFile = readFile.replace(']','', 150) #remove ]
        readFile = readFile.replace('  ',' ', 150) #remove double white space
        readFile = readFile.replace('\n','', 150) #remove new line
        #put in list
        data = [str(x) for x in readFile.split(' ')] #list of strings
        #remove stray elements
        for x in data[:]:
            if(x == ''):
                data.remove(x)
        #convert each element to float
        for s in range(10):
            data[s] = float(data[s])
        self.loadbh2 = [] # LOADED BIASES H2
        count = 0
        for k in range(1):
            column = []
            for i in range(10):
                column.append(data[count])
                count += 1
            self.loadbh2.append(column)

        #OUTPUT LAYER WEIGHTS
        #read, string file
        readFile = open('weightsOutput.txt', 'r').read()
        #process string
        readFile = readFile.replace('[','', 150) #remove [
        readFile = readFile.replace(']','', 150) #remove ]
        readFile = readFile.replace('  ',' ', 150) #remove double white space
        readFile = readFile.replace('\n','', 150) #remove new line
        #put in list
        data = [str(x) for x in readFile.split(' ')] #list of strings
        #remove stray elements
        for x in data[:]:
            if(x == ''):
                data.remove(x)
        #convert each element to float
        for s in range(80):
            data[s] = float(data[s])
        self.loadwout = [] # LOADED WEIGHTS O
        count = 0
        for k in range(10):
            column = []
            for i in range(8):
                column.append(data[count])
                count += 1
            self.loadwout.append(column)

        #OUTPUT LAYER BIASES
        #read, string file
        readFile = open('biasOutput.txt', 'r').read()
        #process string
        readFile = readFile.replace('[','', 150) #remove [
        readFile = readFile.replace(']','', 150) #remove ]
        readFile = readFile.replace('  ',' ', 150) #remove double white space
        readFile = readFile.replace('\n','', 150) #remove new line
        #put in list
        data = [str(x) for x in readFile.split(' ')] #list of strings
        #remove stray elements
        for x in data[:]:
            if(x == ''):
                data.remove(x)
        #convert each element to float
        for s in range(8):
            data[s] = float(data[s])
        self.loadbout = [] # LOADED BIASES O
        count = 0
        for k in range(1):
            column = []
            for i in range(8):
                column.append(data[count])
                count += 1
            self.loadbout.append(column)

    def test(self, testData):
        #function to test network

        start = time.clock()
        #store testData in local list
        self.inData = np.array(testData)
        
        #hidden layer 1
        self.weighted_sum_HL1 = np.dot(self.inData, self.loadwh1)
        self.weighted_sum_HL_withBias1 = self.weighted_sum_HL1 + self.loadbh1 #weighted sum + bias 10x1
        self.hiddenlayer1test = sigmoid(self.weighted_sum_HL_withBias1) #activation function 10x1

        #hidden layer 2
        self.weighted_sum_HL2 = np.dot(self.hiddenlayer1test, self.loadwh2)
        self.weighted_sum_HL_withBias2 = self.weighted_sum_HL2 + self.loadbh2 #weighted sum + bias 10x1
        self.hiddenlayer2test = sigmoid(self.weighted_sum_HL_withBias2) #activation function 10x1

        #output layer
        self.weighted_sum_OL = np.dot(self.hiddenlayer2test, self.loadwout)
        self.weighted_sum_OL_withBias = self.weighted_sum_OL + self.loadbout #weighted sum + bias 8x1
        self.outputtest = sigmoid(self.weighted_sum_OL_withBias) #activation function 8x1

        end = time.clock()
        print(' ')
        print(end - start) #time in seconds
        print(self.outputtest.shape)
        print(self.outputtest)

#DEMO
        
#training data input: 8 sets of x,y,z for 8 gestures
          #[0 0 0 0 0 0 0 0 0 0 0 0]
inPut = [ [0.22, 0.21, 0.23, 0.25, 0.26, 0.25, 0.29, 0.20, 0.21, 0.23, 0.23, 0.22],#g1
          [0.22, 0.21, 0.20, 0.25, 0.18, 0.25, 0.29, 0.27, 0.21, 0.23, 0.23, 0.30],#g1
          [0.22, 0.32, 0.23, 0.37, 0.26, 0.25, 0.41, 0.19, 0.35, 0.23, 0.03, 0.22],#g1
          #[0 0 0 1 1 1 0 0 0 1 1 1]
          [0.02, 0.11, 0.16, 0.87, 0.89, 0.92, 0.12, 0.02, 0.12, 0.90, 0.79, 0.91],#g2
          [0.08, 0.11, 0.16, 0.98, 0.89, 0.91, 0.15, 0.02, 0.12, 0.95, 0.88, 0.91],#g2
          [0.23, 0.11, 0.16, 0.98, 0.89, 0.92, 0.12, 0.02, 0.12, 0.90, 0.88, 0.91],#g2
          #[1 1 1 0 0 0 1 1 1 0 0 0]
          [0.96, 0.99, 0.98, 0.11, 0.12, 0.31, 0.92, 0.87, 0.95, 0.11, 0.30, 0.24],#g3
          [0.98, 0.99, 0.98, 0.11, 0.12, 0.21, 0.97, 0.79, 0.66, 0.13, 0.32, 0.24],#g3
          [0.78, 0.89, 0.90, 0.11, 0.12, 0.31, 0.92, 0.87, 0.59, 0.02, 0.07, 0.20],#g3
          #[1 0 1 0 1 0 1 0 1 0 1 0]
          [0.68, 0.12, 0.98, 0.10, 0.79, 0.21, 0.95, 0.12, 0.85, 0.14, 0.83, 0.11],#g4
          [0.64, 0.15, 0.98, 0.10, 0.87, 0.12, 0.95, 0.12, 0.85, 0.14, 0.83, 0.11],#g4
          [0.80, 0.12, 0.98, 0.10, 0.67, 0.21, 0.95, 0.12, 0.92, 0.14, 0.83, 0.11],#g4
          #[0 1 0 1 0 1 0 1 0 1 0 1]
          [0.12, 0.98, 0.11, 0.90, 0.32, 0.97, 0.08, 0.70, 0.12, 0.89, 0.28, 0.72],#g5
          [0.12, 0.98, 0.11, 0.90, 0.19, 0.86, 0.14, 0.90, 0.12, 0.80, 0.28, 0.75],#g5
          [0.12, 0.80, 0.19, 0.96, 0.32, 0.75, 0.10, 0.93, 0.12, 0.89, 0.10, 0.70],#g5
          #[0 0 1 1 0 0 1 1 0 0 1 1]
          [0.29, 0.11, 0.94, 0.98, 0.29, 0.15, 0.92, 0.86, 0.01, 0.03, 0.90, 0.92],#g6
          [0.17, 0.11, 0.94, 0.98, 0.29, 0.29, 0.92, 0.86, 0.11, 0.03, 0.97, 0.89],#g6
          [0.29, 0.12, 0.96, 0.87, 0.29, 0.10, 0.93, 0.86, 0.01, 0.03, 0.92, 0.72],#g6
          #[1 1 0 0 1 1 0 0 1 1 0 0]
          [0.92, 0.90, 0.12, 0.17, 0.85, 0.80, 0.16, 0.29, 0.94, 0.93, 0.12, 0.19],#g7
          [0.91, 0.95, 0.02, 0.15, 0.90, 0.80, 0.16, 0.19, 0.91, 0.76, 0.12, 0.14],#g7
          [0.92, 0.92, 0.12, 0.10, 0.94, 0.86, 0.13, 0.19, 0.80, 0.92, 0.14, 0.19],#g7
          #[1 1 1 1 1 1 1 1 1 1 1 1]
          [0.99, 0.91, 0.92, 0.93, 0.80, 0.82, 0.99, 0.60, 0.69, 0.72, 0.89, 0.84],#g8
          [0.55, 0.85, 0.90, 0.91, 0.82, 0.82, 0.99, 0.91, 0.69, 0.72, 0.84, 0.84],#g8
          [0.99, 0.91, 0.92, 0.82, 0.70, 0.82, 0.69, 0.92, 0.69, 0.72, 0.75, 0.84] ]#g8

#training data output: 8 sets of output, 1 for each gesture

outPut = [ [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00],#g1
           [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00],#g1
           [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00],#g1
           
           [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00],#g2
           [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00],#g2
           [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00],#g2
           
           [0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00],#g3
           [0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00],#g3
           [0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00],#g3

           [0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00],#g4
           [0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00],#g4
           [0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00],#g4
           
           [0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00],#g5
           [0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00],#g5
           [0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00],#g5
           
           [0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00],#g6
           [0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00],#g6
           [0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00],#g6
           
           [0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],#g7
           [0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],#g7
           [0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],#g7
           
           [1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],#g8
           [1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],#g8
           [1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00] ]#g8

#1st testing data, expected output [0,0,0,0,0,0,0,1]
testMe1 = [ [0.03, 0.12, 0.12, 0.23, 0.07, 0.12, 0.32, 0.29, 0.07, 0.12, 0.20, 0.29] ]
#2nd testing data, expected output [1,0,0,0,0,0,0,0]
testMe2 = [ [0.94, 0.90, 0.89, 0.99, 0.78, 0.99, 0.87, 0.90, 0.89, 0.60, 0.78, 0.54] ]
#3rd testing data, expected output [0,0,1,0,0,0,0,0]
testMe3 = [ [0.12, 0.03, 0.89, 0.99, 0.29, 0.05, 0.87, 0.95, 0.29, 0.18, 0.78, 0.94] ]

#create network
sample = NeuNet()

#train network
#sample.train(inPut, outPut)

#load weights and biases
sample.loadWB()

#test network
print(' ')
print("New input! Expected output should be [0 0 0 0 0 0 0 1]-ish")
sample.test(testMe1)
print(' ')
print("New input! Expected output should be [1 0 0 0 0 0 0 0]-ish")
sample.test(testMe2)
print(' ')
print("New input! Expected output should be [0 0 1 0 0 0 0 0]-ish")
sample.test(testMe3)
