# -*- coding: utf-8 -*-
import os
import random
import datetime
import operator
import slic as sc
import numpy as np
import collections
from PIL import Image
from extreme import StackedELMAutoEncoder, BinaryELMClassifier

def sigmoid(x):
    return 1. / (1 + np.exp(-x))
        
##########################################################
##  ELMTree
##########################################################
class ELMTree(object):
    def __init__(self, radius, sampleSize, numThreshold, numELM, weight, fileName):
        seed = 123
        if radius is None:
            Exception('Error: radius is None.')
        self.radius = radius
        self.sampleSize = sampleSize
        self.numThreshold = numThreshold
        self.np_rng = np.random.RandomState(seed)
        self.fileName = fileName
        self.elm_hidden = numELM
        self.elm_coef = None
        self.weight = weight
        self.dir_name = 'elm/'

    def train(self, train_pic, freq, limit):
        # train decision tree
        print_time('train tree', self.fileName)
        sample = {}
        self.train_pic = train_pic
        for i in xrange(len(train_pic)):
            w,h = train_pic[i].getSize()
            for j in xrange(0, w, freq):
                for k in xrange(0, h, freq):
                    # bootstrap
                    if random.random() < freq:
                        sample[i,j,k] = 0
                    
        s_index = 0
        e_index = 1
        node_list = []
        node_length = 1
        currentDepth = 0
        while s_index < e_index and currentDepth < limit:
            print_time("depth:%d" % (currentDepth), self.fileName)
            currentDepth += 1
            for index in xrange(s_index, e_index):
                forceTerminal = not currentDepth < limit
                data = [list(x) for x in sample.iterkeys() if sample[x] == index]
                isTerminal, param = self.getOptParam(data, train_pic, forceTerminal)
                if isTerminal:
                    node_list.append([isTerminal, param, []])
                if not isTerminal:
                    # inner node
                    l_index, r_index = node_length, node_length + 1
                    l_data, l_label, r_data, r_label = self.divide(data, param, train_pic)
                    for l in l_data:
                        sample[tuple(l)] = l_index
                    for r in r_data:
                        sample[tuple(r)] = r_index
                    node_length += 2
                    node_list.append([isTerminal, param, [l_index, r_index]])
                    
            # update
            s_index = e_index
            e_index = node_length
            
        # initialize input
        print_time('get class distribution', self.fileName)
        input = {}
        for i in xrange(len(train_pic)):
            w,h = train_pic[i].getSize()
            for j in xrange(0, w):
                for k in xrange(0, h):
                    input[i,j,k] = 0

        # get class distribution
        for index, node in enumerate(node_list):
            data = [list(x) for x in input.iterkeys() if input[x] == index]
            isTerminal, param, child = node
            if isTerminal:
                # terminal node
                hist = self.getHist(data, train_pic)
                node_list[index] = [isTerminal, hist, []]
            else:
                # inner node
                l_index, r_index = child
                l_data, l_label, r_data, r_label = self.divide(data, param, train_pic)
                for l in l_data:
                    input[tuple(l)] = l_index
                for r in r_data:
                    input[tuple(r)] = r_index

        # set grown node_list 
        self.node_list = node_list                    

    def test(self, test_pic):
        # initialize input
        input = {}
        for i in xrange(len(test_pic)):
            w,h = test_pic[i].getSize()
            for j in xrange(w):
                for k in xrange(h):
                    input[i,j,k] = 0

        # predict class distribution
        predict = {}
        for index,node in enumerate(self.node_list): 
            data = [list(x) for x in input.iterkeys() if input[x] == index]
            isTerminal, param, child = node
            if isTerminal:
                # terminal node
                for d in data:
                    predict[tuple(d)] = param
            else:
                # inner node
                l_index, r_index = child
                l_data, l_label, r_data, r_label = self.divide(data, param, test_pic)
                for l in l_data:
                    input[tuple(l)] = l_index
                for r in r_data:
                    input[tuple(r)] = r_index
        return predict

    def test_sub(self, input, test_pic):
        # get class distribution
        index = 0
        while True:
            print_time('index:%d' % index, self.fileName)
            node = self.node_list[index]
            isTerminal, param, child = node
            if isTerminal:
                # terminal node
                return param
            else:
                # inner node
                val = self.function(input, param, test_pic)
                isLR = child
                index = isLR[val > 0]
                    
    def divide(self, data, param, data_pic):
        lr_data = [[], []]
        lr_label = [[], []]
        for i,element in enumerate(data):
            index = (self.function(element, param, data_pic) > 0)
            lr_data[index].append(element)
            lr_label[index].append(data_pic[element[0]].getSignal(element[1], element[2]))

        l_data, r_data = lr_data
        l_label, r_label = lr_label
        return l_data, l_label, r_data, r_label

    def function(self, element, param, picture):
        i,x,y = element
        weight, bias, beta = param
        crop = picture[i].cropData(x, y, self.radius)
        hidden = sigmoid(np.dot(weight.T, crop) + bias)
        output = np.dot(beta.T, hidden) # sigmoid(np.dot(hidden, beta))
        return output - 0.5 # constant theta

    def gini(self, l_label, r_label):
        # get gini (minimize)
        g = 0
        set_size = len(l_label) + len(r_label)
        for label in [l_label, r_label]:
            sub_size = len(label)
            counter = collections.Counter(label).most_common()
            for c in counter:
                p = 1. * c[1] / sub_size
                sub = (1. * sub_size / set_size)
                #print "sub", sub * p * (1. - p) 
                g += sub * p * (1. - p)
        return g

    def getLabelList(self, data, data_pic):
        label_list = []
        for element in data:
            i,x,y = element
            label_list.append(data_pic[i].getSignal(x,y))
        return label_list

    def getHist(self, data, data_pic):
        label_list = self.getLabelList(data, data_pic)
        hist = collections.Counter(label_list)
        for h in hist.most_common():
            label_index = h[0]
            hist[label_index] *= self.weight[label_index]        
        return {i:hist[i] for i in xrange(1,9)} # for Etrims8
    
    def getOptParam(self, data, data_pic, forceTerminal):
        # check terminal 
        label = self.getLabelList(data, data_pic)
        if len(set(label)) == 1 or forceTerminal:
            # terminal
            return True, None

        # find optimized parameter
        obj = None
        optParam = None
        for i in xrange(self.numThreshold):
            print_time('th: %i' % i, self.fileName)
            param = self.generate_threshold(data)
            l_data, l_label, r_data, r_label = self.divide(data, param, data_pic)
            g = self.gini(l_label, r_label)
            if len(l_data) == 0 or len(r_data) == 0:
                continue
            if obj is None or g < obj:
                optParam = param
                obj = g
        if optParam is None:
            # terminal
            return True, None
        # inner
        return False, optParam                
        
    def generate_threshold(self, data):
        # crop data
        sample_input, label = [], []
        num = min(len(data), self.sampleSize)
        sample_index = random.sample(data, num)
        for temp in sample_index:
            i,x,y = temp
            sample_input.append(self.train_pic[i].cropData(x, y, self.radius))
            label.append(self.train_pic[i].getSignal(x,y))

        # label
        label_index = []
        numL, numR = 0, 0
        for l in collections.Counter(label).most_common():
            if numL < numR:
                numL += l[1]
                label_index.append(l[0])
            else:
                numR += l[1]

        sample_signal = [1 if l in label_index else 0 for l in label]
            
        # train elm
        elm = BinaryELMClassifier(n_hidden=self.elm_hidden)
        weight, bias, beta = elm.fit(sample_input, sample_signal)
        return weight, bias, beta
            
    def info(self):
        if not self.node_length is None:
            print_time("Information: number of node = %d" % (self.node_length), self.file_name)
        else:
            print_time("Information: self.node_length is not defined", self.file_name)    

##########################################################
##  ELMAETree
##########################################################
class ELMAETree(ELMTree):
    def __init__(self, radius, sampleSize, numThreshold, numELM, weight, fileName):
        seed = 123
        if radius is None:
            Exception('Error: radius is None.')
        self.radius = radius
        self.sampleSize = sampleSize
        self.numThreshold = numThreshold
        self.np_rng = np.random.RandomState(seed)
        self.fileName = fileName
        self.elm_hidden = numELM
        self.elm_coef = None
        self.weight = weight
        self.dir_name = 'elmae/'

    def function(self, element, param, picture):
        i, x, y = element
        selected_dim, theta, betas, biases = param
        crop = self.picture[i].cropData(x, y, self.radius)
        for i, beta in enumerate(betas):
            bias = biases[i]
            crop = sigmoid(np.dot(crop, beta.T) + bias)
        return crop[selected_dim] - theta
        
    def generate_threshold(self, data):
        #print "Generate ", size, " divide functions"
        selmae = StackedELMAutoEncoder(n_hidden=self.elm_hidden, coef=self.elm_coef, visualize=self.visualize)
        sample = []
        num = min(len(data), (2*self.radius+1)*(2*self.radius+1))
        sample_index = random.sample(data, num)
        for temp in sample_index:
            i,x,y = temp
            sample.append(self.picture[i].cropData(x, y, self.radius))
        betas, biases = selmae.fit(sample)

        numpy_data = np.array(selmae.extraction(sample))
        selected_dim = self.np_rng.randint(self.elm_hidden[-1])
        selected_row = numpy_data.T[selected_dim]
        min_row = selected_row.min()
        max_row = selected_row.max()
        theta = self.np_rng.rand() * (max_row - min_row) + min_row
        return selected_dim, theta, betas, biases

##########################################################
##  STTree
##########################################################

class STTree(ELMTree):
    def __init__(self, radius, numThreshold, weight, fileName):
        seed = 123
        if radius is None:
            Exception('Error: radius is None.')
        self.radius = radius
        self.numThreshold = numThreshold
        self.np_rng = np.random.RandomState(seed)
        self.fileName = fileName
        self.weight = weight
        self.func = ['add', 'sub', 'abs', 'uni']
        self.dir_name = 'st/'

    def function(self, element, param, picture):
        i, x, y = element
        f, pos, theta = param
        [x1, y1, c1], [x2, y2, c2] = pos

        if f == 'add':
            return picture[i].getData(x + x1, y + y1)[c1] + picture[i].getData(x + x2, y + y2)[c2] - theta
        if f == 'sub':
            return picture[i].getData(x + x1, y + y1)[c1] - picture[i].getData(x + x2, y + y2)[c2] - theta
        if f == 'abs':
            return abs(picture[i].getData(x + x1, y + y1)[c1] - picture[i].getData(x + x2, y + y2)[c2]) - theta
        if f == 'uni':
            return picture[i].getData(x + x1, y + y1)[c1] - theta
        
    def generate_threshold(self, data):
        f = self.func[random.randint(0, len(self.func)-1)]
        theta = random.random()
        x1, y1, x2, y2 = [random.randint(-1 * self.radius, self.radius) for col in xrange(4)]
        c1, c2 = [random.randint(0, 2) for col in xrange(2)]
        pos = [[x1, y1, c1], [x2, y2, c2]]
        theta = random.random()
        if f == 'add':
            theta = random.random() * 2
        if f == 'sub':
            theta = random.random() * 2 - 1
        if f == 'abs':
            theta = random.random()
        if f == 'uni':
            theta = random.random()        
        return f, pos, theta
        
##########################################################
##  Pic
##########################################################

class Pic(object):
    __slots__ = ['data', 'signal', 'spixel', 'palette',
                 'slength', 'scenter', 'sdic', 'w', 'h']
    def __init__(self, data, signal, spixel):
        self.w, self.h = data.size
        self.palette = signal.getpalette()
        self.setData(data)
        self.setSignal(signal)
        self.setSpixel(spixel)

    def setData(self, data):
        data_list = []
        for x in xrange(self.w):
            temp = []
            for y in xrange(self.h):
                temp.append(list(data.getpixel((x,y))))
            data_list.append(temp)
        self.data = data_list
        
    def setSignal(self, signal):
        signal_list = []
        for x in xrange(self.w):
            temp = []
            for y in xrange(self.h):
                temp.append(signal.getpixel((x,y)))
            signal_list.append(temp)
        self.signal = signal_list

    def setSpixel(self, spixel):
        self.slength = np.max(spixel) + 1
        self.spixel = spixel.T.tolist()

        super_dic = {}
        super_count = np.zeros(self.slength)
        super_label = []
        for i in xrange(self.slength):
            super_label.append([])
        
        super_center = np.zeros((self.slength, 2))
        for x in xrange(self.w):
            for y in xrange(self.h):
                super_center[self.spixel[x][y]] += [x, y]
                super_count[self.spixel[x][y]] += 1
                super_label[self.spixel[x][y]].append(self.getSignal(x,y))
        for i,c in enumerate(super_count):
            super_center[i] /= c
            
        self.scenter = super_center.astype(np.int64).tolist()
        for i in xrange(self.slength):
            super_dic[i] = collections.Counter(super_label[i]).most_common()[0][0]
        self.sdic = super_dic
        
    def getSize(self):
        return self.w, self.h

    def getSSize(self):
        return self.slength

    def getData(self, x, y):
        if x < 0 or x >= self.w:
            # out of x_range
            return [0,0,0]
        if y < 0 or y >= self.h:
            # out of y_range
            return [0,0,0]
        # in range
        return self.data[x][y]

    def getSignal(self, x, y):
        # in range
        return self.signal[x][y]

    def getPalette(self):
        return self.palette        
    
    def getSIndex(self, x, y):
        return self.spixel[x][y]

    def getSData(self, index, dx, dy):
        x,y = self.scenter[index]
        x,y = x+dx, y+dy
        if x < 0 or x >= self.w:
            # out of x_range
            return [0,0,0]
        if y < 0 or y >= self.h:
            # out of y_range
            return [0,0,0]
        # in range
        return self.data[x][y]

    def getSSignal(self, index):
        return self.sdic[index]
    
    def cropData(self, x, y, radius):
        crop = []
        for dx in range(x-radius, x+radius+1):
            for dy in range(y-radius, y+radius+1):
                crop += self.getData(dx, dy)
        crop = (1. * np.array(crop) / 255).tolist()
        return crop

    def cropSData(self, index, radius):
        x,y = self.scenter[index]
        return self.cropData(x,y,radius)
    
##########################################################
##  print
##########################################################
def print_parameter(param, FILE_NAME):
    cmd = 'echo %s >> %s' % (param, FILE_NAME)
    os.system(cmd)
    
def print_time(message, FILE_NAME):
    d = datetime.datetime.today()
    string = '%s/%s/%s %s:%s:%s.%s %s' % (d.year, d.month, d.day, d.hour, d.minute,
                                          d.second, d.microsecond, message)
    cmd = 'echo %s >> %s' % (string, FILE_NAME)
    os.system(cmd)

        
##########################################################
##  load_etrims
##########################################################
    
def load_etrims(radius, size, shuffle, name, n_superpixels, compactness):
    # ----- path initialize -----
    root_path = '../Dataset/etrims-db_v1/'
    an_name = 'annotations/'
    im_name = 'images/'
    et_name = '08_etrims-ds/'
    an_path = root_path + an_name + et_name
    im_path = root_path + im_name + et_name
    dir_list = os.listdir(an_path)
        
    # ----- train index -----
    train_index = []
    DATA_SIZE = size # max=60 
    TRAIN_SIZE = 2 * size / 3 # max=40
    train_index = random.sample(range(DATA_SIZE), TRAIN_SIZE) if shuffle else range(TRAIN_SIZE)

    # ----- test set and train set -----
    train_set = []
    test_set = []
    test_or_train = [test_set, train_set]
    for i in xrange(DATA_SIZE):
        # open annotation.png and image.jpg
        dis = dir_list[i]
        file_name = dis.split(".")[0]
        annot_path = an_path + file_name + ".png"
        annotation = Image.open(annot_path)
        image_path = im_path + file_name + ".jpg"
        image = Image.open(image_path)
        spixel = sc.slic_n(np.array(image), n_superpixels, compactness)

        
        # get index and set picture
        index = i in train_index
        picture = Pic(image, annotation, spixel)
        test_or_train[index].append(picture)

        # print filename
        print_time("eTRIMS: %s" % file_name, name)

    # ----- finish -----
    print_parameter(train_index, name)
    print_time("eTRIMS: train=%d test=%d" % (len(train_set), len(test_set)), name)
    return train_set, test_set

def compute_weight(data_pic):
    # compute label weight from train picture
    label = []
    for i,p in enumerate(data_pic):
        w,h = p.getSize()
        for x in xrange(w):
            for y in xrange(h):
                label.append(data_pic[i].getSignal(x,y))
    label_weight = {}
    for l in collections.Counter(label).most_common():
        label_weight[l[0]] = 1. / l[1]
    return label_weight
        
##########################################################
##  handle forest
##########################################################

def forest_test(forest, test_pic, fileName):

    # ---------- get class distribution (pixel-wise) ----------
    hist = {}
    hist_list = []
    for i,tree in enumerate(forest):
        print_time('tree:%d' % i, fileName)
        hist_list.append(tree.test(test_pic))

    print_time('predict', fileName)       
    for i,p in enumerate(test_pic):
        width, height = p.getSize()
        for j in xrange(width):
            for k in xrange(height):
                hist[i,j,k] = {}
                for c in xrange(1,9):
                    hist[i,j,k][c] = sum(map(lambda x:x[c], map(lambda h:h[i,j,k], hist_list)))

    # ---------- pixel wise ----------
    # """
    # -*- get predict and score -*-
    predict_list, score_list = [], []
    predict, score = predict_pixel(hist, test_pic, fileName)
    for h in hist_list:
        p,s = predict_pixel(h, test_pic, fileName)
        predict_list.append(p)
        score_list.append(s)

    print_time('forest_pixel: %f' % (score), fileName)    
    for i,s in enumerate(score_list):
        print_time('tree%d_pixel: %f' % (i, s), fileName)
        
    print_time('draw_pixel', fileName)    
    draw_pixel(predict, test_pic, "forest_pixel")
    for i,p in enumerate(predict_list):
        draw_pixel(p, test_pic, 'tree%d_pixel' % i)
    # """

    # ---------- super-pixel wise ----------
    # """
    # -*- get predict and score -*-
    predict_list, score_list = [], []
    predict, score = predict_superpixel(hist, test_pic, fileName)
    for h in hist_list:
        p,s = predict_superpixel(h, test_pic, fileName)
        predict_list.append(p)
        score_list.append(s)

    print_time('forest_super: %f' % (score), fileName)    
    for i,s in enumerate(score_list):
        print_time('tree%d_super: %f' % (i, s), fileName)
        
    print_time('draw_super', fileName)    
    draw_superpixel(predict, test_pic, "forest_super")
    for i,p in enumerate(predict_list):
        draw_superpixel(p, test_pic, 'tree%d_super' % i)
    # """

        
def predict_pixel(hist, picture, fileName):
    # ---------- pixel wise ----------
    count = 0
    one_count = 0
    predict = {}
    for i,p in enumerate(picture):
        width, height = p.getSize()
        for j in xrange(width):
            for k in xrange(height):
                label = picture[i].getSignal(j,k)
                # predict & count
                predict[i,j,k] = max(hist[i,j,k].iteritems(), key=operator.itemgetter(1))[0]
                if predict[i,j,k] == label:
                    one_count += 1
        print_time('%dth picture: %d (%d)' % (i, one_count, width * height), fileName)
        count += one_count
        one_count = 0
    length = len(predict)
    return predict, (1. * count / length)

def draw_pixel(predict, picture, file_name):
    # ---------- pixel wise ----------
    for i,p in enumerate(picture):
        w,h = p.getSize()
        image = Image.new('P', (w,h))
        image.putpalette(p.getPalette())
        for j in xrange(w):
            for k in xrange(h):
                image.putpixel((j,k), predict[i,j,k])
        name = file_name + str(i) + ".png"
        image.save(name)
    
def predict_superpixel(hist, picture, fileName):
    # ---------- super-pixel wise ----------
    # compress hist
    length = 0
    super_hist = {}
    for i,p in enumerate(picture):
        width, height = p.getSize()
        length += (width * height)
        for index in xrange(p.getSSize()):
            super_hist[i,index] = {col:0 for col in xrange(1,9)}         
        for j in xrange(width):
            for k in xrange(height):
                index = p.getSIndex(j,k)
                for col in xrange(1,9):
                    super_hist[i,index][col] = super_hist[i,index][col] + hist[i,j,k][col] 

    # predict
    predict = {}
    for i,p in enumerate(picture):
        for index in xrange(p.getSSize()):
            predict[i,index] = max(super_hist[i,index].iteritems(), key=operator.itemgetter(1))[0]

    # count
    count = 0
    one_count = 0
    for i,p in enumerate(picture):
        width, height = p.getSize()
        for j in xrange(width):
            for k in xrange(height):
                label = p.getSignal(j,k)
                index = p.getSIndex(j,k)
                if predict[i,index] == label:
                    one_count += 1
        print_time('%dth picture: %d (%d)' % (i, one_count, width * height), fileName)
        count += one_count
        one_count = 0
    return predict, (1. * count / length)    
                    
def draw_superpixel(predict, picture, file_name):
    # ---------- super-pixel wise ----------
    for i,p in enumerate(picture):
        w,h = p.getSize()
        image = Image.new('P', (w,h))
        image.putpalette(p.getPalette())
        for j in xrange(w):
            for k in xrange(h):
                index = p.getSIndex(j,k)
                image.putpixel((j,k), predict[i,index])
        name = file_name + str(i) + ".png"
        image.save(name)

def do_forest(boxSize, dataSize, unShuffle, sampleFreq,
              isELMF, isELMAEF, isSTF,
              dataPerTree, depthLimit, numThreshold, numTree, sampleSize,
              numHidden,
              n_superpixels, compactness,
              fileName):
    
    # ----- initialize -----
    print_time('eTRIMS: init', fileName)

    radius = (boxSize - 1) / 2
    train_pic, test_pic = load_etrims(radius=radius, size=dataSize,
                                      shuffle=not unShuffle, name=fileName,
                                      n_superpixels=n_superpixels, compactness=compactness)

    print_parameter([boxSize, dataSize, unShuffle, sampleFreq], fileName)
    print_parameter([isELMF], fileName)
    print_parameter([dataPerTree, depthLimit, numThreshold, numTree], fileName)
    print_parameter([numHidden], fileName)
    print_time('eTRIMS: radius=%d, depth_limit=%s, data_size=%d, num_func=%d'
               % (radius, str(depthLimit), dataSize, numThreshold), fileName)

    # compute label weight
    weight = compute_weight(train_pic)
    
    if isELMF:
        print_time('ELM forest', fileName)
        print_time('init', fileName)
        forest = []
        print_time('train', fileName)
        for i in xrange(numTree):
            print_time('tree: %i' % i, fileName)
            tree = ELMTree(radius=radius, sampleSize=sampleSize, numThreshold=numThreshold,
                           numELM = numHidden, weight=weight, fileName=fileName)        
            tree.train(train_pic=train_pic, freq=sampleFreq, limit=depthLimit)
            forest.append(tree)

        print_time('test', fileName)
        forest_test(forest, test_pic, fileName)

    if isELMAEF:
        print_time('ELM-AE forest', fileName)
        print_time('init', fileName)
        forest = []
        print_time('train', fileName)
        for i in xrange(numTree):
            print_time('tree: %i' % i, fileName)
            tree = ELMAETree(radius=radius, sampleSize=sampleSize,
                             numThreshold=numThreshold,
                             numELM = numHidden, weight=weight, fileName=fileName)        
            tree.train(train_pic=train_pic, freq=sampleFreq, limit=depthLimit)
            forest.append(tree)

        print_time('test', fileName)
        forest_test(forest, test_pic, fileName)

    if isSTF:
        print_time('Semantic Texton forest', fileName)
        print_time('init', fileName)
        forest = []
        print_time('train', fileName)
        for i in xrange(numTree):
            print_time('tree: %i' % i, fileName)
            tree = STTree(radius=radius,
                          numThreshold=numThreshold,
                          weight=weight, fileName=fileName)        
            tree.train(train_pic=train_pic, freq=sampleFreq, limit=depthLimit)
            forest.append(tree)

        print_time('test', fileName)
        forest_test(forest, test_pic, fileName)


    # ----- finish -----
    print_time('eTRIMS: finish', fileName)

    
