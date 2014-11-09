import numpy as np
import sys,csv

class Perceptron:
    def __init__(self,trainingDataFilename,testDataFilename,numIters,learnRate):
        self.trainFile = trainingDataFilename
        self.testFile = testDataFilename
        self.Iters = numIters
        self.n = learnRate
        self.w = []  # list of weights
        self.b = 0   # bias
        self.x = []  # data table
        self.tX = [] # test data
        self.y = []  # outputs
        self.tY = [] # test outputs
        self.operations()

    # Logic
    def operations(self):
        self.readData()     # read in the data
        self.learnModel()   # learn the model
        self.loss01()       # calculate 0/1 loss

    # Learn the Perceptron Model
    def learnModel(self):
        yHat, e = 0, 0
        for it in range(0,self.Iters):
            for i in range(0,len(self.x)):
                yHat = np.sign(np.dot(self.w,self.x[i])+self.b)
                if(yHat != self.y[i]):
                    e = self.y[i] * self.n
                    for j in range(0,len(self.w)):
                        self.w[j] = self.w[j] + (e * self.x[i][j])
                    # endfor
                    self.b = self.b + e
                # endif
            # endfor
        # endfor

    # 0/1 Loss Calculation
    def loss01(self):
        yHat = 0
        wrong = 0
        for i in range(0,len(self.tX)):
            yHat = np.sign(np.dot(self.w,self.tX[i]))
            if(yHat != self.tY[i]):
                wrong += 1
            # endif
        # endfor
        ratio = float(wrong) / float(len(self.tX))
        print("ZERO-ONE LOSS = %f"%(ratio))
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
