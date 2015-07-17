# -*- coding: utf-8 -*-
import gradient_boosting

# config data
boxSize = 15# 15

dataSize = 60
unShuffle = False
sampleFreq = 5 # 1

# config Tree Type
isREG = True
isELMREG = False

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
fileName = "g_test.log"

if __name__ == '__main__':
    gradient_boosting.do_forest(boxSize, dataSize, unShuffle, sampleFreq,
                                isREG, isELMREG,
                                dataPerTree, depthLimit, numThreshold, numTree, sampleSize,
                                numHidden,
                                n_superpixels, compactness,
                          fileName)
