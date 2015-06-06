# -*- coding: utf-8 -*-
import semantic4rest

# config data
boxSize = 5# 15

dataSize = 6
unShuffle = True
sampleFreq = 4 # 1

# config Tree Type
isELMF = True
isELMAEF = True
isSTF = True

# config forest
dataPerTree = 0.5
depthLimit = 10
numThreshold = 4 # 400
numTree = 5
sampleSize = boxSize * boxSize * 3

# config ELMF
numHidden = boxSize * boxSize * 3 * 2

# config slic
n_superpixels = 500
compactness = 10

# config fileName
fileName = "s_test.log"

if __name__ == '__main__':
    semantic4rest.do_forest(boxSize, dataSize, unShuffle, sampleFreq,
                            isELMF, isELMAEF, isSTF,
                            dataPerTree, depthLimit, numThreshold, numTree, sampleSize,
                            numHidden,
                            n_superpixels, compactness,
                            fileName)
