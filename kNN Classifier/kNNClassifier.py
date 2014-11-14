import numpy as np
import sys,csv,math

class kNNClassifier:
    def __init__(self,trainingDataFilename,testDataFilename,k):
        self.trainFile = trainingDataFilename
        self.testFile = testDataFilename
        self.k = k   # The radius we want to use
        self.x = []  # Attributes
        self.y = []  # Classes
        self.tX = [] # Test attributes
        self.tY = [] # Test classes
        self.operations()

    # Logic
    def operations(self):
        self.readData()
        print(self.tX)
        print(self.tY)
        self.predictClasses()

    # Predict the classes of the test data
    def predictClasses(self):
        correct = 0    # Number of correctly predicted classes
        for i in range(0,len(self.tX)):
            
        print("I accurately predicted %d object classes!"%(correct))
        # endfor

    # Calculate Distance
    def calcDistance(self,objectNum):
        distance = 0
        classLabel = ""
        for i in range(0,len(self.x)):
            
        return {distance,classLabel}

    # Read the .CSV files
    def readData(self):
        f = open(self.trainFile,'rt')
        reader = csv.reader(f)
        unparsedTrimmed = []
        for row in reader: # Read in contents
            unparsedTrimmed.append(row)
        # endfor
        f.close()
        for row in range(0,len(unparsedTrimmed)):
            self.x.append(unparsedTrimmed[row][:-1])
            self.x[row] = list(map(float,self.x[row]))
            self.y.append(unparsedTrimmed[row][-1])
        # endfor

        f = open(self.testFile,'rt')
        reader = csv.reader(f)
        unparsedTrimmed = []
        for row in reader: # Read in contents
            unparsedTrimmed.append(row)
        # endfor
        f.close()
        for row in range(0,len(unparsedTrimmed)):
            self.tX.append(unparsedTrimmed[row][:-1])
            self.tX[row] = list(map(float,self.tX[row]))
            self.tY.append(unparsedTrimmed[row][-1])
        # endfor
