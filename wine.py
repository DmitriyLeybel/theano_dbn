from DeepLearningTutorialsmaster.code import DBN
import numpy as np
import cPickle, gzip, numpy as np


# Load the dataset
# f = gzip.open('mnist.pkl.gz')
# train_set, valid_set, test_set = cPickle.load(f)
# f.close()

# matplot not in environment||used for testing
# import matplotlib.pyplot as plt
# train_set.shape = ((train_set.size)//500, 500)
# plt.imshow(train_set[0])
# plt.show()

# Data preparation
wineArray = np.loadtxt('wine.data',dtype=float,delimiter=',')
wineLabels = wineArray[:,0]
winePred = wineArray[:,1:]

trainNum = int((50/70.0)*winePred.shape[0])
train_set = (winePred[0:trainNum, :], wineLabels[0:trainNum])

validNum = int((10/70.0)*winePred.shape[0])
testNum = int((10/70.0)*winePred.shape[0])
valid_set = (winePred[trainNum:trainNum+validNum,:],
             wineLabels[trainNum:trainNum+validNum])

test_set = (winePred[trainNum+validNum:trainNum+validNum+testNum,:],
            wineLabels[trainNum+validNum:trainNum+validNum+testNum])

listOfSets = [train_set,valid_set,test_set]
fileP = gzip.open('wines.pklz','wb')
cPickle.dump(listOfSets,fileP)
fileP.close()
fileP = gzip.open('wines.pklz','r')
fileP.close()


DBN.test_DBN(dataset='wines.pklz')