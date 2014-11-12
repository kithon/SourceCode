# coding: utf-8
# online learning
                                  
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
    
    def __init__(self, n_input, n_hidden, n_output, activation=None):
        # initialize size of neuron
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        if activation is None:
            activation = sigmoid
        self.activation = activation

        # initialize auto_encoder
        auto_encoders = []
        for num in n_hidden:
            ae = ELMAutoEncoder(n_hidden=num)
            auto_encoders.append(ae)
        self.auto_encoders = auto_encoders

    def pre_train(self, input):
        # pre_train
        print "pre_train"
        data = input
        betas = []
        for ae in self.auto_encoders:
            print "ae fitting"
            ae.fit(data)

            # Path 1
            
            # part 1
            #beta = ae.get_beta()

            # part 2
            beta = ae.get_beta()
            #beta = beta / np.linalg.norm(beta)
            
            """
            darkness : 
            """
            
            # Path 2

            # part 1 use activation and bias
            data = self.activation(np.dot(data, beta.T) + ae.get_bias()) ###test

            # part 2 dot data and beta only
            #data = np.dot(data, beta.T)

            betas.append(beta)

        self.betas = betas
        self.data4fine = data

    def fine_tune(self, teacher):
        print "fine_tune"
        # initialize classes
        classes = []
        for t in teacher:
            if not t in classes:
                classes.append(t)
        self.classes = classes
        self.n_output = len(self.classes)

        # initialize signal
        signal = []
        id_matrix = np.identity(self.n_output).tolist()
        for t in teacher:
            signal.append(id_matrix[self.classes.index(t)])

        # initialize data
        data = self.data4fine  # data = self.activation(self.data4fine)

        # set beta
        H = np.matrix(data)
        Hp = H.I
        #Hp = H.T * (H * H.T).I
        beta = np.dot(Hp, np.array(signal))
        self.fine_beta = beta
        
        
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
        # fit
        self.pre_train(input)
        self.fine_tune(teacher)

    def predict(self, input):
        hidden = self.pre_extraction(input)
        output = self.fine_extraction(hidden)
        output = np.array(output)
        
        predict_classes = []
        for o in output:
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
                 c=0., n_hidden=50, seed=123, domain=[-1., 1.]):
        # initialize
        self.activation = activation
        self.c = c
        self.n_hidden = n_hidden
        self.np_rng = np.random.RandomState(seed)
        self.domain = domain
        
    def construct(self, input):
        # set parameter of layer
        self.input = input
        self.n_input = len(input[0])
        self.n_output = len(input[0])
        low, high = self.domain

        # set weight and bias (randomly)
        weight = self.np_rng.uniform(low = low, high = high, size = (self.n_input, self.n_hidden))
        bias = self.np_rng.uniform(low = low, high = high, size = self.n_hidden)

        # orthogonal weight and forcely regularization
        """
        for i in xrange(len(weight)):
            w = weight[i]
            for j in xrange(0,i):
                w = w - weight[j].dot(w) * weight[j]
            w = w / np.linalg.norm(w)
            weight[i] = w
        """

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
                           self.c)

        
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

    def error(self, input):
        # get error
        pre = self.predict(input)
        err = pre - input
        err = err * err
        print "sum of err^2", err.sum()
        return err.sum()

    def get_weight(self):
        return self.weight

    def get_bias(self):
        return self.bias

    def get_beta(self):
        return self.layer.beta
    

class ELMClassifier(object):
    """
    Extreme Learning Machine
    
    
    """

    def __init__(self, activation=sigmoid, vector='orthogonal', regular=True,
                 c=0., n_hidden=50, seed=123, domain=[-1., 1.]):
        # initialize
        self.activation = activation
        self.vector = vector
        self.regular = regular
        self.c = c
        self.n_hidden = n_hidden
        self.np_rng = np.random.RandomState(seed)
        self.domain = domain
        
    def construct(self, input, teacher):
        # set parameter of layer
        self.input = input
        self.teacher = teacher
        classes = []
        for t in teacher:
            if not t in classes:
                classes.append(t)
        self.classes = classes
        self.n_input = len(input[0])
        self.n_output = len(self.classes)
        low, high = self.domain

        # set weight and bias (randomly)
        weight = self.np_rng.uniform(low = low, high = high, size = (self.n_input, self.n_hidden))
        bias = self.np_rng.uniform(low = low, high = high, size = self.n_hidden)

        if self.vector == 'orthogonal':
            # orthogonal weight and forcely regularization
            print "set weight and bias orthogonaly"
            for i in xrange(len(weight)):
                w = weight[i]
                for j in xrange(0,i):
                    w = w - weight[j].dot(w) * weight[j]
                w = w / np.linalg.norm(w)
                weight[i] = w

            if self.regular:
                # bias regularization
                denom = np.linalg.norm(bias)
                if denom != 0:
                    denom = bias / denom
            
            
        elif self.vector == 'random':
            # randomly and regulatization
            print "set weight and bias randomly"
            if self.regular:
                #for i,w enumerate(weight.T):
                for i,w in enumerate(weight):
                    denom = np.linalg.norm(w)
                    if denom != 0:
                        #weight.T[i] = w / denom
                        weight[i] = w / denom

                # bias regularization
                denom = np.linalg.norm(bias)
                if denom != 0:
                    bias = bias / denom
            
        else:
            print "warning: vector isn't orthogonal or random"
            
        
        # set weight and bias
        self.weight = weight
        self.bias = bias     
        
            
        # initialize layer
        self.layer = Layer(self.activation,
                           [self.n_input, self.n_hidden, self.n_output],
                           self.weight,
                           self.bias,
                           self.c)

        
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

    def get_weight(self):
        return self.weight

    def get_bias(self):
        return self.bias

    def get_beta(self):
        return self.layer.beta

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
        # activation from input to hidden
        return self.activation(np.dot(self.w.T, input) + self.b)

    def get_h2o(self, hidden):
        # activation from hidden to output
        return np.dot(self.beta.T, hidden)

    def get_output(self, input):
        # activation from input to output
        hidden = self.get_i2h(input)
        output = self.get_h2o(hidden)
        return output
    
    def fit(self, input, signal):
        # set self.beta from activation
        H = []
        for i in input:
            H.append(self.get_i2h(i))
        H = np.matrix(H)
        if self.c == 0:
            # Tend to be Memory Error
            #Hp = H.T * (H * H.T).I
            Hp = H.I
        else:
            id_matrix = np.matrix(np.identity(len(input)))
            Hp = H.T * ((id_matrix / (self.c * 1.)) + H * H.T).I            
        Hp = np.array(Hp)
        self.beta = np.dot(Hp, np.array(signal))


if __name__ == "__main__":
    
    train = [[1, 1], [2, 2], [-1, -1], [-2, -2]]
    label = [1, 1, -1, -1]
    test = [[3, 3], [-3, -3]]

    model = MLELMClassifier(n_input=2, n_hidden=[4,8,5], n_output=1)

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
