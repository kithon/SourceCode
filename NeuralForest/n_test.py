# -*- coding: utf-8 -*-
import neural4rest

# config data
boxSize = 15# 15

dataSize = 60
unShuffle = False
sampleFreq = 5 # 1

# config Tree Type
isELMF = True
isELMAEF = False
isSTF = False

# config forest
dataPerTree = 0.25
depthLimit = None
numThreshold = 1 # 400
numTree = 5
sampleSize = boxSize * boxSize * 3

# config ELMF
numHidden = boxSize * boxSize * 3 * 2

# config slic
n_superpixels = 500
compactness = 10

# config fileName
fileName = "n_test.log"

if __name__ == '__main__':
    neural4rest.do_forest(boxSize, dataSize, unShuffle, sampleFreq,
                          isELMF, isELMAEF, isSTF,
                          dataPerTree, depthLimit, numThreshold, numTree, sampleSize,
                          numHidden,
                          n_superpixels, compactness,
                          fileName)
