# -*- coding: utf-8 -*-
import semantic4rest

# config data
boxSize = 15# 15

dataSize = 6
unShuffle = False
sampleFreq = 4 # 1

# config Tree Type
isELMF = True

# config forest
dataPerTree = 0.5
depthLimit = 10
numThreshold = 4 # 400
numTree = 5

# config ELMF
numHidden = boxSize * boxSize * 3 * 2

# config slic
n_superpixels = 500
compactness = 10

# config fileName
fileName = "s_test.log"

if __name__ == '__main__':
    semantic4rest.do_forest(boxSize, dataSize, unShuffle, sampleFreq,
                            isELMF,
                            dataPerTree, depthLimit, numThreshold, numTree,
                            numHidden,
                            n_superpixels, compactness,
                            fileName)
