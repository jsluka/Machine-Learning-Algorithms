from collections import Counter as ct
from random import shuffle
import numpy as np
import scipy.stats as ss
import sys,csv,math

##########################################################################
# decisionTree:                                                          #
#   The class responsible for the construction and testing of a decision #
#   tree. Takes as an input the training data and the test data (which   #
#   must be given in .csv files.                                         #
#                                                                        #
##########################################################################
class decisionTree:
    def __init__(self,trainingDataFilename,testDataFilename):
        self.trainingFile = trainingDataFilename
        self.testFile = testDataFilename
        self.thres = 3.841          #threshold
        self.x = []                 #all binary data
        self.features = []
        self.testX = []             #all binary data
        self.y = []                 #actual answers
        self.testY = []             #test actual answers
        self.headNode = None        #this is the head of the actual model
        self.totalNodes = 0 
        self.predictions = []       #the predicted outcomes of our test data
        self.loss = 0
        self.operations()

    # Primary logic (don't put this in __init__, dummy)
    def operations(self):
        self.readData()             #read in the data
        self.convertToBinary()      #convert data to binary from continuous
        self.buildTreeHead()        #create the head of the tree
        self.predictTuples()        #make predictions using our model and the test data
        self.zeroOneLoss()          #compute our 0/1 loss
        return self.loss

    #####################
    # TREE CONSTRUCTION #
    #####################

    # Creates the head of the tree
    def buildTreeHead(self):
        maxChi, feature = self.calculateChis(self.x,self.y)
        self.headNode = node(feature,self.x,self.y)
        self.addNodes(self.headNode)
        print("Total number of nodes in tree: %d"%(self.totalNodes))

    # Add the nodes recursively
    def addNodes(self,split):
        zeros,ones = 0,0
        for i in range(0,len(split.expected)):
            if split.expected[i] == 0: zeros+=1
            else: ones+=1
        if zeros > ones: split.prediction = 0
        else: split.prediction = 1
        self.totalNodes += 1
        sample0, sample1, expected0, expected1 = self.trimProperResponses(split.sample,split.expected,split.feature)
        chi0, fea0 = self.calculateChis(sample0,expected0)
        chi1, fea1 = self.calculateChis(sample1,expected1)
        if chi0 >= 3.481:
            split.node_0 = node(fea0,sample0,expected0)
            self.addNodes(split.node_0)
        if chi1 >= 3.481:
            split.node_1 = node(fea1,sample1,expected1)
            self.addNodes(split.node_1)

    #####################
    # DATA MANIPULATION #
    #####################

    # Split the sample across a feature
    def trimProperResponses(self,sample,expected,featureNum):
        sample0, sample1 = [], []
        expected0, expected1 = [], []
        correct = 0
        for i in range(0,len(sample)):
            if sample[i][featureNum] == 0:
                expected0.append(expected[i])
                sample0.append(sample[i])
            elif sample[i][featureNum] == 1:
                expected1.append(expected[i])
                sample1.append(sample[i])
        for i in range(0,len(sample0)):
            sample0[i].pop(featureNum)
        for i in range(0,len(sample1)):
            sample1[i].pop(featureNum)
        return sample0, sample1, expected0, expected1

    # Calculate chi scores
    def calculateChis(self,sample,expected):
        chis,a,b,c,d = [],0,0,0,0
        for j in range(0,len(sample[0])):
            for i in range(0,len(sample)):
                if sample[i][j] == 1:
                    if expected[i] == 1: a+=1
                    else: b+=1
                elif sample[i][j] == 0:
                    if expected[i] == 1: c+=1
                    else: d+=1
            numerator=(((a*d)-(b*c))**2)*(a+b+c+d)
            denomenator=(a+b)*(c+d)*(b+d)*(a+c)
            if denomenator == 0:
                chis.append(-1)
            else:
                chis.append((numerator/denomenator))
            a,b,c,d = 0,0,0,0
        return max(chis), chis.index(max(chis))

    ##################
    # TEST FUNCTIONS #
    ##################
    
    # Iterate through test tuples and make predictions
    def predictTuples(self):
        for i in range(0,len(self.testX)):
            tup = self.testX[i]
            val = self.walkTree(tup,self.headNode)
            self.predictions.append(val)

    # Walks the tree given for a given tuple
    def walkTree(self,tup,node):
        currentStep = node
        while(True):
            if tup[currentStep.feature] == 0:
                if currentStep.node_0 != None:
                    currentStep = currentStep.node_0
                else:
                    return currentStep.prediction
            elif tup[currentStep.feature] == 1:
                if currentStep.node_1 != None:
                    currentStep = currentStep.node_1
                else:
                    return currentStep.prediction

    # Compute the zero/one loss
    def zeroOneLoss(self):
        wrong = 0
        for i in range(0,len(self.testY)):
            if(self.testY[i] != self.predictions[i]):
                wrong+=1
        ratio = float(wrong) / float(len(self.testY))
        print("ZERO-ONE LOSS: %f"%(ratio))
        self.loss = ratio

    ######################
    # FILE IO/PROCESSING #
    ######################
    
    # Converts continuous values to binary
    def convertToBinary(self):
        medians = []
        for i in range(0,len(self.x[0])):
            tempList = []
            for j in range(0,len(self.x)):
                tempList.append(self.x[j][i])
            medians.append(np.median(tempList))
        for i in range(0,len(self.x[0])):
            self.features.append([])
            for j in range(0,len(self.x)):
                if self.x[j][i] > medians[i]:
                    self.x[j][i] = 1
                    self.features[i].append(1)
                else:
                    self.x[j][i] = 0
                    self.features[i].append(0)
        for i in range(0,len(self.y)):
            if self.y[i] == -1:
                self.y[i] = 0
        for row in range(0,len(self.x)): 
            self.x[row] = list(map(int,self.x[row]))
        #end of the training data    
        medians = []
        for i in range(0,len(self.testX[0])):
            tempList = []
            for j in range(0,len(self.testX)):
                tempList.append(self.testX[j][i])
            medians.append(np.median(tempList))
        for i in range(0,len(self.testX[0])):
            self.features.append([])
            for j in range(0,len(self.testX)):
                if self.testX[j][i] > medians[i]:
                    self.testX[j][i] = 1
                    self.features[i].append(1)
                else:
                    self.testX[j][i] = 0
                    self.features[i].append(0)
        for i in range(0,len(self.testY)):
            if self.testY[i] == -1:
                self.testY[i] = 0
        for row in range(0,len(self.testX)): 
            self.testX[row] = list(map(int,self.testX[row]))
        #end of the test data
        print("Converted all values to binary")
        

    # Read Data (as CSV file)
    def readData(self):
        f = open(self.trainingFile,'rt')
        reader = csv.reader(f)
        unparsedTrimmed = []
        for row in reader: # Read in contents
            unparsedTrimmed.append(row)
        f.close()
        for row in range(0,len(unparsedTrimmed)): # Trim rows, store output seperately
            self.x.append(unparsedTrimmed[row][:-1])
            self.x[row] = list(map(float,self.x[row]))
            self.y.append(unparsedTrimmed[row][-1])
        self.y = list(map(int,self.y)) # Convert from string to float
        #end of the training data
        f = open(self.testFile,'rt')
        reader = csv.reader(f)
        unparsedTrimmed = []
        for row in reader:
            unparsedTrimmed.append(row)
        f.close()
        for row in range(0,len(unparsedTrimmed)):
            self.testX.append(unparsedTrimmed[row][:-1])
            self.testX[row] = list(map(float,self.testX[row]))
            self.testY.append(unparsedTrimmed[row][-1])
        self.testY = list(map(int,self.testY))
        #end of the test data
        print("Data files read")

##########################################################################
# Node:                                                                  #
#   Represents a single decision in a decision tree. Takes as input the  #
#   feature on which we split, the given sample (a reduced tuple), and   #
#   the given expected list (again reduced).                             #
#                                                                        #
#   Contains pointers to the false and true nodes, as well as the current#
#   prediction for the given decision.                                   #
##########################################################################
class node:
    def __init__(self,feature,sample,expected):
        self.feature = feature
        self.sample = sample
        self.expected = expected
        self.prediction = 0
        self.node_0 = None
        self.node_1 = None
