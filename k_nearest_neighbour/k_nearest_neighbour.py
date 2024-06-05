import numpy as np
class knn:
    def fit(self,x,y,k=3):
        self.x=x
        self.y=y
        self.k=k
        
    def dist(self,a,b): # to find the distance betwen the data and the group
        return np.sum(np.square(a-b),axis=1)
    
    def neighbours(self,xin):  # to fing the nearest neighbours
        distance = self.dist(self.x,xin)
        t=np.argsort(distance,axis=0)
        neigh=self.y[t]
        neigh=neigh[:self.k,0]
        cla_count=np.bincount(neigh.astype(int))
        return np.argmax(cla_count)
    
    def normalize(self,x): # to fit the data more accurately
        mean = np.mean(x,axis=0)
        std = np.std(x,axis=0)
        return (x-mean)/std
    
    def predict(self,xin):
        m=np.shape(xin)[0]
        y_pred=np.zeros((m,1))
        for i in range(m):
            y_pred[i] = self.neighbours(xin[i,:])
        return y_pred
    
    def accuracy(self,x_test,y_test):
        return np.mean(self.predict(x_test)==y_test)*100