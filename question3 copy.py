import pandas 
import numpy as np
from math import sqrt , exp, ceil

def sigmoid(array):
    return (1/(1+np.exp(-array)))

class NeuralNetwork :
    def __init__(self , hiddenWeights, HiddenBiases, WeightsOutput, BiasesOutput, learningRate, momentum):  
        self.weightsHidden = hiddenWeights
        self.BiasesHidden = HiddenBiases
        self.weightsOutput = WeightsOutput
        self.biasesOutput = BiasesOutput
        self.lr = learningRate
        self.momentum = momentum

    def train(self , inputs , targ):
        TransIn = inputs.T
        targets= targ.T

        HiddenIn =  np.dot(self.weightsHidden,TransIn) 
        # i = 0
        # for i in range(len(HiddenIn)): 
        #     HiddenIn[i] += self.BiasesHidden
        #     i +=1

        HiddenOut = sigmoid(HiddenIn)#output for hidden layer

        finalIn = np.dot(self.weightsOutput,HiddenOut)# + self.biasesOutput
        finalOut = sigmoid(finalIn)#output for final layer
       

        #commmence backPropagation
        #            derivative of act func         Error
        OutputErr =  finalOut*(1 -finalOut ) *(targets - finalOut)

        #            output weights*output error to find how much that node affects it * derivative of activation function
        HiddenErr = np.dot(self.weightsOutput.T, OutputErr) * HiddenOut *(1-HiddenOut)

        #                                       sigma output times weights going into Output
        self.weightsOutput += self.lr * np.dot(OutputErr ,np.transpose(HiddenOut))
        

        self.weightsHidden += self.lr * np.dot(HiddenErr , np.transpose(TransIn))


    def test(self , inputs) :
        TransIn = inputs.T

        HiddenIn =  np.dot(self.weightsHidden,TransIn) 

        HiddenOut = sigmoid(HiddenIn)#output for hidden layer

        finalIn = np.dot(self.weightsOutput,HiddenOut)# + self.biasesOutput
        finalOut = sigmoid(finalIn)#output for final layer
        return finalOut  
        
    
 


if __name__ == "__main__": 
    url = "q3_Dataset.csv"
    #load training data
    names = ["Dose1", "Dose2", "Node1", "Node2"]
    dataset = pandas.read_csv(url, names=names)
    array = dataset.values

    #Load tranning data
    descriptive = array[:,0:2]
    target = array[:,2:]
    alpha = np.array([7.1655,6.9060,2.0033,6.1144,5.9538])
    W_o = 0.3074

    #load test data
    url = "q3Test_Dataset.csv"
    names = ["Dose1", "Dose2", "Node1", "Node2"]
    testDataset = pandas.read_csv(url, names=names)
    testArray = testDataset.values
    testDescrip = testArray[:,0:2]
    testTarget = testArray[:,2]


    inputNodes = 2 
    OutputNodes = 2
    hiddenNodes = 4
    epoch = 5


    hiddenWeights= np.random.normal(0,pow(hiddenNodes,-0.5), (hiddenNodes,inputNodes )) 
    outputWeights =np.random.normal(0,pow(OutputNodes,-0.5), (OutputNodes,hiddenNodes ))

    HiddenBiases  = np.empty([hiddenNodes])
    OutputBiases = np.empty([OutputNodes])
    
    HiddenBiases.fill(1)
    OutputBiases.fill(1)

    learningRate = 0.5
    momentum = 1 

    NN = NeuralNetwork(hiddenWeights,HiddenBiases,outputWeights,OutputBiases, learningRate,momentum)

    for i in range(epoch):
        NN.train(descriptive,target)
    predictions = NN.test(testDescrip)
    n = len(predictions)
    i =0 
    x= 0 
    LabelArr = list()
    for val in predictions: 
        print(val)
        if (val[0] > val[1]):
            label = 0.99
            
        else : 
            label = 0.01
        LabelArr.append()
        temp = testTarget[i]
        if label == temp:
            x +=1 
        i +=1 

    print("accuracy = %f" %(x/n)) 