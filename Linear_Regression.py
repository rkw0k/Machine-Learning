"""
Given an input data of (x_i, y_i), this program computes the linear regression.
Using Gradient Descent and initial parameter values, the program iterates
until a predetermined final step. It outputs the coefficient of determination 
of the final iteration revealing how well the line fits the data.
Ricky Kwok, rickyk9487@gmail.com, 2014-10-08
"""

import numpy as np
import matplotlib.pyplot as plt

class LinReg(object):
    
    def __init__(self, data):
        """ initialize input data """
        self._data = data
        # A row vector to store parameters
        self.params = np.ones((1,data.shape[1]))
        return
    
    def data_x(self):
        """ Creates a matrix of x-values and ones. """
        matrix = np.ones(self._data.shape)
        matrix[:,1] = self._data[:,0]
        return matrix
    
    def model(self):
        """ Computes the function in the model. """
        theta = self.params
        x = self.data_x()
        h = np.dot(theta, x.T)
        return h
    
    def cost(self):
        """ Computes the cost. """
        m = data.shape[0]
        v = self.model() - self._data[:,1]
        C = ((2*m) ** (-1)) * np.dot(v,v.T)
        return C

    def update(self, alpha):
        """ Update the parameters based on a given alpha. Does not return."""
        m = data.shape[0]
        x = self.data_x()
        v = self.model() - self._data[:,1]
        Ctheta = (m ** (-1)) * np.dot(x.T,v.T)
        self.params = self.params - alpha*Ctheta.T

    def coeff(self):
        """ Returns the coefficient of determination. """
        y = self._data[:,1]
        ybar = sum(y)/len(y)
        h = self.model()
        SS_tot = np.dot( (y-ybar).T, (y-ybar))
        SS_res = np.dot( (y-h), (y-h).T)
        R_squared = 1-SS_res/SS_tot        
        return R_squared
        
    def const(self):
        alpha = 0.001
        n_iter = 5000
        return alpha, n_iter
        
def show_data(theta, data, const, R_squared):
    """ Prints or plots data points, regression line, coeff of determ."""
    print 'For %d iterations with alpha = %f, ' %(const[1], const[0])
    print 'the coefficient of determination is %f.' %(R_squared)
    
    def f(t):
        return theta[1]*t+theta[0]
          
    X = data[:,0]
    endpts = [min(X), max(X)]
    dist = (endpts[1]-endpts[0]) * 0.25
    t1 = np.arange(endpts[0]-dist, endpts[1]+dist, 0.01)

    plt.close()
    plt.plot(t1,f(t1))
    plt.scatter(data[:,0], data[:,1])
    plt.show()                        
                                                                        
if __name__ == "__main__":
    """ Calls class LinReg to run Gradient Descent."""
    data = np.genfromtxt("ex1data1.txt", delimiter = ",")
    #data = np.genfromtxt("filename", delimiter = ",")
    linear = LinReg(data)
    const = linear.const()
    
    def run_model():
        for i in xrange(const[1]):
            linear.update(const[0])
    
    test = run_model()
    cost = linear.cost()
    theta = linear.params[0]
    R_squared = linear.coeff()
    print theta
    show_data(theta,data, const, R_squared)
    