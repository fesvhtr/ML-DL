import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# cost function
def cost(ctheta,x,y):
    co = np.sum(np.power(x * ctheta.T - y, 2))
    m = x.shape[0]
    return co/(2*m)
# gradient descent
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


# input the data
data = pd.read_csv("ex1data1.txt", names=["population", "profit"])
print("shape of data:{}".format(data.shape))
data.head()
data.insert(0,"intercept term",1)

# matrix the data
X = np.mat(data.iloc[:,:2].values)
Y = np.mat(data.iloc[:,2:3].values)
# get the shape of X and set zeros for the origin theta
theta = np.mat(np.zeros(X.shape[1]))
iterTimes = 1500
alpha = 0.01
finalCost,finalTheta = grad(theta,iterTimes,alpha,X,Y)


# result by normal equation
print(np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y))
print(finalTheta[0,0])
# plot data1
org_axis = data.plot.scatter(x='population',y='profit',color ='Blue',label = 'oriData')
# plot Regression function
xR = np.linspace(-10,10,100)
yR = finalTheta[0,0] + finalTheta[0,1] * xR
plt.figure(num=2)
plt.plot(xR,yR)
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))
# plot cost function
plt.figure(num=3)
plt.plot(finalCost)
plt.show()