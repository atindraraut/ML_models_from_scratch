from tkinter import Y
import numpy as np

class Logit_Reg():

    def __init__(self,learning_rate,iterations):
        self.learning_rate = learning_rate
        self.iterations = iterations


    def fit(self,X,Y):
        self.m ,self.n  = X.shape

        self.w = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y

        #using gradient descent algorithm for optimisation
        for i in range(self.iterations):
            self.update_weights()
        return self

    def update_weights(self):

        #writing the sigmoid function
        y_hat = 1/(1 + np.exp(-(self.X.dot(self.w) + self.b)))

        #defining the derivatives of log loss /cost  function
        dw = (1/self.m)*np.dot(self.X.T,(y_hat - self.Y))
        db = (1/self.m)*np.sum(y_hat-self.Y)

        #updating weight and bias
        self.w = self.w -(self.learning_rate*dw)
        self.b = self.b - (self.learning_rate*db)
        
    def predict(self,X):
        y_pred = 1/(1 + np.exp(-(X.dot(self.w) + self.b)))
        y_pred = np.where( y_pred > 0.5, 1, 0)
        return y_pred
    
class linear_reg():

    def __init__(self,learning_rate,iterations):
        self.learning_rate = learning_rate
        self.iterations = iterations

    def fit(self,x,y):
        self.m,self.n = x.shape

        self.w = np.zeros(self.n)
        self.b = 0
        self.X = x
        self.Y = y
    
        #using the gradient descent algorithm for optimisation
        for i in range(self.iterations):
            self.update_weights()
        return self

    def update_weights(self):
        
        y_hat = np.dot(self.X,self.w) + self.b

        #defining the derivatives of loss/cost function
        dw = (1/self.m)*np.dot(self.X.T,(y_hat-self.Y))
        db = (1/self.m)*np.sum(y_hat-self.Y)

        self.w = self.w - (self.learning_rate*dw)
        self.b = self.b - (self.learning_rate*db)

    def predict(self,X):
        return np.dot(X,self.w) + self.b
    