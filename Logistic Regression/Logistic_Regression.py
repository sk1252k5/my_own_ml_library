import numpy as np

class log_reg:
    
    def fit(self,x,y,lambda_=0):
        self.x = x
        self.y = y
        self.w = np.zeros((np.shape(x)[1],1))
        self.b = [0]
        self.lambda_ = lambda_
        
    def normalize(self,x):
        Mean=np.mean(x,axis=0)
        Std=np.std(x,axis=0)
        return (x-Mean)/Std,Mean,Std
    
    def sigmoid(self,x):
        return 1/(1 + np.exp(-x))
    
    def cost_fn(self):
        hypo = self.sigmoid(self.x.dot(self.w)+self.b)
        regu = self.lambda_/2/np.size(self.y)*np.square(self.x)
        return np.sum(-1 / np.size(self.y) * (np.log(hypo).T.dot(self.y) + np.log(1-hypo).T.dot(1-self.y))+regu)
    
    def gradiant_descent(self,l_rate,no_of_iter,lambda_=0):
        m=np.size(self.y)
        for i in range(no_of_iter):
            hypo=self.sigmoid(self.x.dot(self.w)+self.b)
            dif=hypo-self.y     
            self.w-=1/m*self.x.T.dot(dif) + lambda_/m*self.w
            self.b-=np.sum(1/m*dif)
        return self.w,self.b
    
    def predict(self,xin):
        return self.sigmoid(np.dot(xin,self.w)+self.b) >= 0.5
    
    def accuracy(self,x_test,y_test):
        acc=np.mean(self.predict(x_test)==y_test)
        return acc*100
