import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def cost(ctheta,x,y):
    co = np.sum(np.power(x * ctheta.T - y, 2))
    m = x.shape[0]
    return co/(2*m)

def grad(gtheta,iter,alpha,x,y):
    m = x.shape[0]
    co = np.zeros(iter)
    temp = np.mat(np.zeros(gtheta.shape[1]))
    for i in range(iter):
        difference = x * gtheta.T - y
        for j in range(gtheta.ravel().shape[1]):
            temp[0,j] = temp[0,j] - (alpha/m) * np.sum(np.multiply(difference,x[:,j]))
            # print(temp)
        gtheta = temp
        co[i] = cost(gtheta,x,y)
    return co, gtheta

data = pd.read_csv('ex1data2.txt',names = ['size','roomNum','price'])
X = data.iloc[:,0:2]
Y = data.iloc[:,2]

# standardize
X = (X-np.mean(X))/np.std(X)
Y = (Y-np.mean(Y))/np.std(Y)
X.insert(0,"Ones",1)
x = np.mat(X.values)
y = np.mat(Y.values)
# define
theta = np.mat(np.zeros(x.shape[1]))
iter = 1500
alpha = 0.01
print(np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y.T))
finalCost,finalTheta = grad(theta,iter,alpha,x,y.T)
print(finalTheta)
plt.figure()
plt.plot(finalCost)
plt.show()
