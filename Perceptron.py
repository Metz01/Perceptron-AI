import numpy as np

class Perceptron:
    
    """
    ---------
    #*PARAMETERS (Given by outside)
        eta : float   #? Learning rate (between 0.0 and 1.0), float
        n_iter : int  #? Number of times the perceptron have to pass over the training dataset, int
        random_state: int   #? Seed for the random number generator used for random weight initialization, int

    #*ATTRIBUTES (From the class)
        w_ : 1d-array   #? Weights after fitting
        errors_ : list  #? Numbers of missclassification in each epoch
    ----------
    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self, X, y):
        """
        ---------
        #* PARAMETERS
            X : {array-like}, shape = [n_example, n_features]
                #? Training Vector, where n_examples is the number of examples and n_features is the number of features
            y : array-like, shape = [n_examples]
                #? Target values, the ones we expect the perceptron to give us (solutions)

        #* RETURNS
            self : object
        -----------
        """

        rgen = np.random.RandomState(self.random_state)     #Create a Random generator with the seed random_state
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1+X.shape[1])   #Create a vector of dimension size of float number > 0 
                                                                        #taken randomly in a gaussian graph with the center in 0.0(loc) and with a standard deviation of 0.01(scale)
                                                                        #note that in a Gaussian 99,7% of the values have to stay between (center-3*standard_dev) and (center+3*standard_dev)
        self.errors_ = []   #? In each cell is stored the number of errors for each epoche

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):            #? zip(a,b) create n tuples with the i-th elementh of each array a=[1,2] b=[3,4] zip(a,b)=[(1,3), (1,4)]
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self, X):
        """CALCULATE NET INPUT"""
        return np.dot(X, self.w_[1:]) + self.w_[0]        #? dot products of two matrix
    
    def predict(self, X):
        """RETURN CLASS LABEL AFTER UNIT STEP"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)    #? where(condition, T, F) if condition is true return T else F