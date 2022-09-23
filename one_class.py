# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 13:08:19 2022

@author: WeiYanPEH
"""

# Generate and plot a synthetic imbalanced classification dataset
from collections import Counter
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from numpy import where
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.svm import OneClassSVM
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor



plt.close('all')        




# define dataset
X, y = make_classification(n_samples=10000,
                           n_features=2, 
                           n_redundant=0,
	                       n_clusters_per_class=1, 
                           weights=[0.999], 
                           flip_y=0, 
                           random_state=4)

# summarize class distribution
counter = Counter(y)
print(counter)
# scatter plot of examples by class label
for label, _ in counter.items():
	row_ix = where(y == label)[0]
	plt.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
plt.title('Distribution of Data')
plt.legend()
plt.show()


y[y == 1] = -1
y[y == 0] = 1


(trainX, testX, 
 trainy, testy) = train_test_split(X, y, 
                                   test_size=0.5, 
                                   random_state=2, 
                                   stratify=y)

trainX = trainX[trainy==1]


# X - some data in 2dimensional np.array
h=0.01
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))


fig, axes = plt.subplots(2, 2,
                         sharex="col", 
                         sharey="row",
                         figsize=(8, 6))

# here "model" is your model's prediction (classification) function
def plot_db(model, ax, title):
    cdict = {-1: 'blue', 1: 'red'}
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]) 
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=plt.cm.Paired)

    for g in [1,-1]:
        ix = np.where(y == g)
        if g==1:
            alpha=0.05
        else:
            alpha = 1
        
        ax.scatter(X[ix, 0], X[ix, 1],
                     label=str(g),
                     c = cdict[g],
                     alpha = alpha,
                     marker = 'x'
                     )
    ax.set_title(title)



# one-class svm for imbalanced binary classification
model = OneClassSVM(gamma='scale', nu=0.01)
model.fit(trainX)
yhat = model.predict(testX)

# calculate score
CM = confusion_matrix(testy, yhat)
print('\nOne Class SVM')
print(CM)

plot_db(model, axes[0,0], 'One Class SVM')



# define outlier detection model
model = IsolationForest()
model.fit(trainX)
yhat = model.predict(testX)

# calculate score
CM = confusion_matrix(testy, yhat)
print('\nIsolation Forest')
print(CM)

plot_db(model, axes[0,1], 'Isolation Forest')



# define outlier detection model
model = EllipticEnvelope()
model.fit(trainX)
yhat = model.predict(testX)

# calculate score
CM = confusion_matrix(testy, yhat)
print('\nElliptic Envelope')
print(CM)

plot_db(model, axes[1,0], 'Elliptic Envelope')




# define outlier detection model
model = LocalOutlierFactor(contamination=0.01, novelty=True)
model.fit(trainX)
yhat = model.predict(testX)

# calculate score
CM = confusion_matrix(testy, yhat)
print('\nLocal Outlier Factor')
print(CM)

plot_db(model, axes[1,1], 'Local Outlier Factor')









