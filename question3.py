import pandas 
import numpy as np
from math import sqrt , exp, ceil

def sigmoid(array):
    return (1/(1+np.exp(-array)))

class NeuralNetwork :
    def __init__(self ,newModel,  hiddenWeights=None, HiddenBiases=None, WeightsOutput=None, BiasesOutput=None, learningRate=None, momentum=None):  
        if newModel : 
            self.weightsHidden = hiddenWeights
            self.BiasesHidden = HiddenBiases
            self.weightsOutput = WeightsOutput
            self.biasesOutput = BiasesOutput
            self.lr = learningRate
            self.momentum = momentum
        else : 
            self.weightsHidden = np.load("./weightsInHidden.npy")
            self.BiasesHidden = np.load("./BiasesHidden.npy")
            self.weightsOutput =  np.load("./weightsHiddenOut.npy")
            self.biasesOutput =  np.load("./BiasesOut.npy")

    def train(self , inputs , targ):
        TransIn = np.transpose(np.array(inputs,ndmin=2))
        targets= np.transpose(np.array(targ,ndmin=2))


        HiddenIn =  np.dot(self.weightsHidden,TransIn) + self.BiasesHidden
        

        HiddenOut = sigmoid(HiddenIn)#output for hidden layer

        
        finalIn = np.dot(self.weightsOutput,HiddenOut)+ self.biasesOutput
        finalOut = sigmoid(finalIn)#output for final layer
       

        #commmence backPropagation
        #            derivative of act func         Error
        OutputErr =  finalOut*(1 -finalOut ) *(targets - finalOut)
        
        #            output weights*output error to find how much that node affects it * derivative of activation function
        HiddenErr = np.dot(np.transpose(self.weightsOutput), OutputErr) * HiddenOut *(1-HiddenOut)

        #                                       sigma output times weights going into Output
        self.weightsOutput += self.lr * np.dot(OutputErr ,np.transpose(HiddenOut))
        

        self.weightsHidden += self.lr * np.dot(HiddenErr , np.transpose(TransIn))

        self.biasesOutput += self.lr * OutputErr
        self.BiasesHidden += self.lr * +HiddenErr


    def test(self , inputs) :
        TransIn = np.transpose(np.array(inputs,ndmin=2))

        HiddenIn =  np.dot(self.weightsHidden,TransIn) 

        HiddenOut = sigmoid(HiddenIn)#output for hidden layer

        finalIn = np.dot(self.weightsOutput,HiddenOut)# + self.biasesOutput
        finalOut = sigmoid(finalIn)#output for final layer
        return finalOut  
        
    
    def SaveModel(self): 
        np.save("./weightsInHidden", self.weightsHidden)
        np.save("./BiasesHidden", self.BiasesHidden)
        np.save("./weightsHiddenOut", self.weightsOutput)
        np.save("./BiasesOut", self.biasesOutput)


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
    epoch = 35
    useNewModel = False

    
    if useNewModel : 
        hiddenWeights= np.random.normal(0,pow(hiddenNodes,-0.5), (hiddenNodes,inputNodes )) 
        outputWeights =np.random.normal(0,pow(OutputNodes,-0.5), (OutputNodes,hiddenNodes ))

        HiddenBiases  = np.random.normal(0, 0.5, (hiddenNodes,1))
        OutputBiases = np.random.normal(0, 0.5, (OutputNodes,1))
        

        learningRate = 0.5
        momentum = 1 

        NN = NeuralNetwork(useNewModel, hiddenWeights,HiddenBiases,outputWeights,OutputBiases, learningRate,momentum)
        for j in range(epoch):
            for i in range(len(descriptive)): 
                NN.train(descriptive[i],target[i])
            
    else : 

        NN = NeuralNetwork(useNewModel)

    x =0 
    n = len(testTarget)
    i=0
    for row in testDescrip:
        predictions = NN.test(row)
        print(predictions)
        
        if (predictions[0] > predictions[1]):
            label = 0.99
            
        else : 
            label = 0.01
        temp = testTarget[i]
        if label == temp:
            x +=1 
        i +=1 
    accuracy = x/n
    print("accuracy = %f" %(accuracy)) 
    


    if accuracy == 1.0 and useNewModel: 
        print("---ModelSaved----")
        NN.SaveModel()