# coding: utf-8
# online learning

import sys
import numpy as np

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def sign(x):
    return (np.sign(x - 0.5) + 1) / 2

class MLELMClassifier(object):
    """
    Multi-Layer Extreme Learning Machine

    
    """
    
    def __init__(self, n_hidden=None, n_coef=None,
                 fine_coef=1000.,activation=None):
        # initialize size of neuron
        if n_hidden is None:
            raise Exception("nlist_hidden is udefined")
        self.n_hidden = n_hidden
        if n_coef is None:
            n_coef = [0] * len(n_hidden)
        self.coef = n_coef
        self.fine_coef = fine_coef
        if activation is None:
            activation = sigmoid
        self.activation = activation

        # initialize auto_encoder
        auto_encoders = []
        for i, num in enumerate(n_hidden):
            ae = ELMAutoEncoder(activation=activation,
                                n_hidden=num, coef=n_coef[i])
            auto_encoders.append(ae)
        self.auto_encoders = auto_encoders

    def pre_train(self, input):
        # pre_train
        print "pre_train"
        data = input
        betas = []
        for i, ae in enumerate(self.auto_encoders):
            # fit auto_encoder
            print " ", i,"ae fit"
            ae.fit(data)

            # get beta
            beta = ae.get_beta()
            
            # part use activation and bias
            act = np.dot(data, beta.T) + ae.get_bias()
            data = self.activation(act)

            # append beta
            betas.append(beta)

        # set betas and data for fine_tune
        self.betas = betas
        self.data4fine = data

    def fine_tune(self, teacher):
        print "fine_tune"
        # get data for fine_tune
        sys.stdout.write("\r  data for fine_tune")
        sys.stdout.flush()
        H = np.array(self.data4fine)
        # data = self.activation(self.data4fine)
        signal = self.signal
        print " done."
        
        # coefficient of regularization for fine_tune
        sys.stdout.write("\r  coefficient")
        sys.stdout.flush()
        np_id = np.identity(min(H.shape))
        if self.fine_coef == 0:
            coefficient = 0
        else:
            coefficient = 1. / self.fine_coef
        print " done."
        
        # pseudo inverse
        sys.stdout.write("\r  pseudo inverse")
        sys.stdout.flush()
        regular = coefficient * np_id
        if H.shape[0] < H.shape[1]:
            Hp = np.linalg.inv(np.dot(H, H.T) + regular)
            Hp = np.dot(H.T, Hp)
        else:
            Hp = np.linalg.inv(np.dot(H.T, H) + regular)
            Hp = np.dot(Hp, H.T)
        print " done."

        # set beta for fine_tune
        sys.stdout.write("\r  set beta")
        sys.stdout.flush()
        beta = np.dot(Hp, np.array(signal))
        self.fine_beta = beta
        print " done."
        
    def pre_extraction(self, input):
        # pre_extraction
        data = input
        for i, ae in enumerate(self.auto_encoders):
            beta = self.betas[i]
            #print "i:", i
            #print "data:", data
            #print "beta:", beta
            data = self.activation(np.dot(data, beta.T) + ae.get_bias())
        return data

    def fine_extraction(self, data):
        #print "fine_extraction"
        #print "data:", np.array(data).shape
        #print "beta:", np.array(self.fine_beta).shape
        return np.dot(data, self.fine_beta)
        
    def fit(self, input, teacher):
        # initialize classes
        classes = []
        for t in teacher:
            if not t in classes:
                classes.append(t)
        self.classes = classes
        self.n_input = len(input[0])
        self.n_output = len(self.classes)
        
        # initialize signal
        signal = []
        id_matrix = np.identity(self.n_output).tolist()
        for t in teacher:
            signal.append(id_matrix[self.classes.index(t)])
        self.signal = signal
            
        # pre_train fine_tune
        self.pre_train(input)
        self.fine_tune(teacher)

    def predict(self, input):
        # get predict_output
        hidden = self.pre_extraction(input)
        output = self.fine_extraction(hidden)
        predict_output = np.array(output)

        # get predict_classes from index of max_function(predict_output)
        predict_classes = []
        for o in predict_output:
            predict_classes.append(self.classes[np.argmax(o)])

        return predict_classes
    
    def score(self, input, teacher):
        # get score
        count = 0
        length = len(teacher)
        predict_classes = self.predict(input)
        for i in xrange(length):
            if predict_classes[i] == teacher[i]:
                count += 1
        return count * 1.0 / length

class ELMAutoEncoder(object):
    """
    Extreme Learning Machine Auto Encoder
    
    
    """

    def __init__(self, activation=sigmoid,
                 n_hidden=50, coef=0., seed=123, domain=[-1., 1.]):
        # initialize
        self.activation = activation
        self.n_hidden = n_hidden
        self.coef = coef
        self.np_rng = np.random.RandomState(seed)
        self.domain = domain
        
    def get_weight(self):
        return self.weight

    def get_bias(self):
        return self.bias

    def get_beta(self):
        return self.layer.beta
            
    def construct(self, input):
        # set parameter of layer
        self.input = input
        self.n_input = len(input[0])
        self.n_output = len(input[0])
        low, high = self.domain

        # set weight and bias (randomly)
        weight = self.np_rng.uniform(low = low,
                                     high = high,
                                     size = (self.n_input,
                                             self.n_hidden))
        bias = self.np_rng.uniform(low = low,
                                   high = high,
                                   size = self.n_hidden)

        # orthogonal weight and forcely regularization
        
        for i in xrange(len(weight)):
            w = weight[i]
            for j in xrange(0,i):
                w = w - weight[j].dot(w) * weight[j]
            w = w / np.linalg.norm(w)
            weight[i] = w
            

        # bias regularization
        denom = np.linalg.norm(bias)
        if denom != 0:
            denom = bias / denom
        
        # set weight and bias
        self.weight = weight
        self.bias = bias     
        
            
        # initialize layer
        self.layer = Layer(self.activation,
                           [self.n_input, self.n_hidden, self.n_output],
                           self.weight,
                           self.bias,
                           self.coef)

        
    def fit(self, input):
        # construct layer
        self.construct(input)

        # fit layer
        self.layer.fit(input, input)
        
    def predict(self, input):
        # get predict_output
        predict_output = []
        for i in input:
            o = self.layer.get_output(i).tolist()
            predict_output.append(o)
        return predict_output
    
    def score(self, input, teacher):
        # get score 
        count = 0
        length = len(teacher)
        predict_classes = self.predict(input)
        for i in xrange(length):
            if predict_classes[i] == teacher[i]: count += 1
        return count * 1.0 / length
    
    def error(self, input):
        # get error
        pre = self.predict(input)
        err = pre - input
        err = err * err
        print "sum of err^2", err.sum()
        return err.sum()
    

class ELMClassifier(object):
    """
    Extreme Learning Machine
    
    
    """

    def __init__(self, activation=sigmoid, vector='orthogonal',
                 coef=0., n_hidden=50, seed=123, domain=[-1., 1.]):
        # initialize
        self.activation = activation
        self.vector = vector
        self.coef = coef
        self.n_hidden = n_hidden
        self.np_rng = np.random.RandomState(seed)
        self.domain = domain
        
    def get_weight(self):
        return self.weight

    def get_bias(self):
        return self.bias

    def get_beta(self):
        return self.layer.beta
    
    def construct(self, input, teacher):
        # set input, teacher and class
        self.input = input
        self.teacher = teacher
        classes = []
        for t in teacher:
            if not t in classes:
                classes.append(t)
        self.classes = classes
        self.n_input = len(input[0])
        self.n_output = len(self.classes)

        # weight and bias
        low, high = self.domain
        weight = self.np_rng.uniform(low = low,
                                     high = high,
                                     size = (self.n_input,
                                             self.n_hidden))
        bias = self.np_rng.uniform(low = low,
                                   high = high,
                                   size = self.n_hidden)

        # condition : orthogonal random else
        if self.vector == 'orthogonal':
            print "set weight and bias orthogonaly"
            for i in xrange(len(weight)):
                w = weight[i]
                for j in xrange(0,i):
                    w = w - weight[j].dot(w) * weight[j]
                w = w / np.linalg.norm(w)
                weight[i] = w

            # regularize bias
            denom = np.linalg.norm(bias)
            if denom != 0:
                denom = bias / denom
                    
        elif self.vector == 'random':
            print "set weight and bias randomly"
            # regularize weight
            
            #for i,w enumerate(weight.T):
            for i,w in enumerate(weight):
                denom = np.linalg.norm(w)
                if denom != 0:
                    #weight.T[i] = w / denom
                    weight[i] = w / denom

            # regularize bias
            denom = np.linalg.norm(bias)
            if denom != 0:
                bias = bias / denom
                    
        else:
            print "warning: vector isn't orthogonal or random"
            
        
        # self weight and bias
        self.weight = weight
        self.bias = bias
            
        # self layer
        self.layer = Layer(self.activation,
                           [self.n_input, self.n_hidden, self.n_output],
                           self.weight,
                           self.bias,
                           self.coef)

        
    def fit(self, input, teacher):
        # construct layer
        self.construct(input, teacher)

        # convert teacher to signal
        signal = []
        id_matrix = np.identity(self.n_output).tolist()
        for t in teacher:
            signal.append(id_matrix[self.classes.index(t)])

        # fit layer
        self.layer.fit(input, signal)
        
    def predict(self, input):
        # get predict_output
        predict_output = []
        for i in input:
            o = self.layer.get_output(i).tolist()
            predict_output.append(o)
        #print "outputs", predict_output

        # get predict_classes from index of max_function(predict_output) 
        predict_classes = []
        for o in predict_output:
            predict_classes.append(self.classes[o.index(max(o))])
        #print "predict" predict_classes

        return predict_classes

    def score(self, input, teacher):
        # get score 
        count = 0
        length = len(teacher)
        predict_classes = self.predict(input)
        for i in xrange(length):
            if predict_classes[i] == teacher[i]: count += 1
        return count * 1.0 / length
    

class Layer(object):
    def __init__(self, activation, size, w, b, c):
        # initialize 
        self.activation = activation
        self.n_input, self.n_hidden, self.n_output = size
        self.c = c
        self.w = w
        self.b = b
        self.beta = np.zeros([self.n_hidden,
                              self.n_output])

    def get_beta(self):
        return self.beta
        
    def get_i2h(self, input):
        return self.activation(np.dot(self.w.T, input) + self.b)

    def get_h2o(self, hidden):
        return np.dot(self.beta.T, hidden)

    def get_output(self, input):
        hidden = self.get_i2h(input)  # from input to hidden
        output = self.get_h2o(hidden) # from hidden to output
        return output
    
    def fit(self, input, signal):
        """
        fit : set beta
        risk : memory error (dot)
        """
        
        # get activation of hidden layer
        H = []
        for i, d in enumerate(input):
            sys.stdout.write("\r    input %d" % (i+1))
            sys.stdout.flush()
            H.append(self.get_i2h(d))
        print " done."

        # coefficient of regularization
        sys.stdout.write("\r    coefficient")
        sys.stdout.flush()
        np_id = np.identity(min(np.array(H).shape))
        if self.c == 0:
            coefficient = 0
        else:
            coefficient = 1. / self.c
        print " done."

        # pseudo inverse
        sys.stdout.write("\r    pseudo inverse")
        sys.stdout.flush()
        H = np.array(H)
        regular = coefficient * np_id
        if H.shape[0] < H.shape[1]:
            Hp = np.linalg.inv(np.dot(H, H.T) + regular)
            Hp = np.dot(H.T, Hp)
        else:
            Hp = np.linalg.inv(np.dot(H.T, H) + regular)
            Hp = np.dot(Hp, H.T)
        print " done."
            
        # set beta
        sys.stdout.write("\r    set beta")
        sys.stdout.flush()
        self.beta = np.dot(Hp, np.array(signal))
        print " done."

if __name__ == "__main__":
    
    train = [[1, 1], [2, 2], [-1, -1], [-2, -2]]
    label = [1, 1, -1, -1]
    test = [[3, 3], [-3, -3]]

    model = MLELMClassifier(n_hidden=[4,8,5])

    model.fit(train, label)

    print model.predict(train)
    print model.predict(test)
    print "score:", model.score(train, label)
    """
    train = [[1, 1], [2, 2], [-1, -1], [-2, -2]]
    label = [1, 1, -1, -1]
    test = [[3, 3], [-3, -3]]
    
    model = ELMClassifier()
    model.fit(train, label)
    pre = model.predict(test)
    print pre
    print model.score(train, label)
    """
