# -*- coding: utf-8 -*-
import semantic4rest

# config data
boxSize = 15
dataSize = 60
unShuffle = True
sampleFreq = 4 # 1

# config Tree Type
isELMF = True

# config forest
dataPerTree = 0.25
depthLimit = 10
numThreshold = 400
numTree = 5

# config ELMF
numHidden = boxSize * boxSize * 3 * 2

# config fileName
fileName = "semantic.log"

if __name__ == '__main__':
    semantic4rest.do_forest(boxSize, dataSize, unShuffle, sampleFreq,
                            isELMF,
                            dataPerTree, depthLimit, numThreshold, numTree,
                            numHidden,
                            fileName)
