import numpy as np
import sys,csv,math

class linearRegression:
    def __init__(self,trainingDataFilename,testDataFilename,numIters,learnRate):
        self.trainFile = trainingDataFilename
        self.testFile = testDataFilename
        self.Iters = numIters
        self.n = learnRate
        self.w = []   # list of weights
        self.b = 0    # bias
        self.x = []   # data table
        self.tX = []  # test data
        self.y = []   # outputs
        self.tY = []  # test outputs
        np.seterr(divide="ignore",invalid="ignore")
        self.operations()

    # Logic
    def operations(self):
        self.readData()    # read in the data
        self.learnModel()  # learn the model
        self.squaredLossFin() # test the model

    # Learn the Regression Model
    def learnModel(self):
        tW = []
        for i in range(0,10):
            for j in range(0,len(self.w)):
                summed = self.sumDot()
                tW.append(self.w[j]-(self.n*(1/len(self.x))*summed*self.x[i][j]))
        for i in range(0,len(self.w)):
            self.w[i] = tW[i]

    def sumDot(self):
        total = 0
        for i in range(0,len(self.x)):
            total = total + np.dot(self.x[i],self.w) - self.y[i]
        return total
            
    # Test the Regression Model
    def squaredLoss(self):
        yHat = 0
        squaredSum = 0
        for i in range(0,len(self.x)):
            squaredSum += np.dot(self.w,self.x[i])**2
        # endfor
        ratio = float(squaredSum) / float(len(self.x))
        return ratio

    # Test the Regression Model
    def squaredLossFin(self):
        yHat = 0
        squaredSum = 0
        for i in range(0,len(self.tX)):
            squaredSum += np.dot(self.w,self.tX[i])**2
        # endfor
        ratio = float(squaredSum) / float(len(self.tX))
        print("SQUARED-LOSS: %f"%ratio)
        print("Learning is fun!")

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
        self.y = list(map(float,self.y))
        
        for x in range(0,len(self.x[0])):
            self.w.append(0)
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
        self.tY = list(map(float,self.tY))
