import os
from typing import List
from matplotlib import markers
from matplotlib import colors
from matplotlib.colors import ListedColormap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Perceptron

s = os.path.join('https://archive.ics.uci.edu', 'ml', 'machine-learning-databases', 'iris', 'iris.data')

#df = pd.read_csv(s, header= None, encoding='utf-8')            #Online
df = pd.read_csv('iris.data', header= None, encoding='utf-8')   #Offline

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

X = df.iloc[0:100, [0,2]].values

plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()

ppn = Perceptron.Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()



def plot_decision_region(X, y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))]) #? Maps a color of the array colors with a unique element of the tag array y

    x1_min, x1_max = X[:, 0].min() - 1, X[:,0].max() +1
    x2_min, x2_max = X[:, 1].min() - 1, X[:,1].max() +1
    xx1, xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution), np.arange(x2_min,x2_max,resolution)) #? Arange return evenly spaced(resolution) values between a cenrtain range(x1_min/max)
                                                            #? Meshgrid(a,b) create a matrix M where M[i,j] is  b[i]+a[j]
    Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)   #? Ravel return a contiguos flattered array (a=([1,2],[3]) a.ravel()=([1,2,3]))
                                    #? classifier.predict(..) use the predict method of the perceptron in every point (spaced by resolution) of the grid
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1,xx2, Z, alpha=0.3, cmap = cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x = X[y == cl, 0], y = X[y == cl, 1], alpha=0.8, c=colors[idx], marker=markers[idx], label=cl, edgecolor='black')


plot_decision_region(X, y, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()