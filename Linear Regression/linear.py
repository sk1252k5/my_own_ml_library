
import numpy as np

#class that models linear regression
class lin_reg:
    def fit(self,x,y,lamda_=0):
        self.x=np.array(x)
        self.y=y
        self.w = np.zeros((np.shape(x)[1],1)) # weight or slope of regression line
        self.b=[0] # y_intercept
        self.lamda_ = lamda_  # regularization factor
    
    def normalize(self,x_in):   # Used to fit the data more accurately
        Mean=np.mean(x_in,axis=0)
        Std=np.std(x_in,axis=0)
        return (x_in - Mean) / Std,Mean,Std
    
    def cost_func(self):        # Mean Square Error
        return np.sum(np.square(self.x.dot(self.w)+self.b-self.y))/2/np.size(self.y)+lambda_/2/np.size(self.y)*np.square(self.w)

    def Gradiant_descent(self,l_rate,no_of_iter,lambda_=0):  #gradient descent
        m=np.size(self.y)
        for i in range(no_of_iter):
            dif=self.x.dot(self.w)+self.b-self.y
            self.w -= l_rate / m * self.x.T.dot(dif) + lambda_ / m * self.w
            self.b -= np.sum(l_rate / m * dif)
        return self.w,self.b

    def predict(self,x_in): # To predict for new values
        return np.dot(x_in,self.w)+self.b
    
    def accuracy(self,x,y):   # To tests accuracy of the predicted data
        y_pred=self.predict(x)
        eror = abs(np.mean((y-y_pred)/y))
        return (1-eror)*100
    