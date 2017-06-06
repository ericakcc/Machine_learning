# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 19:07:22 2017

@author: ericakcc
"""

import numpy as np
import matplotlib.pyplot as plt

X = np.zeros([1,4])
y = np.zeros(1)
for i in range(1,18):
    data = np.load(r'D:\gradddddddddddddddddd\newdata\data_%i.npy' % i)
    for j in range(len(data)):
        if data[j,4] < 2500:
            X = np.append(X,np.array([np.hstack((data[j,2:5],data[j,6]))]),axis=0)
#            X = np.append(X,np.array([data[j,3:5]]),axis=0)
            y = np.append(y,data[j,5])
#X3 = X[:,0] - X[:,1]
#X = np.concatenate((X, np.array([X3]).T), axis=1)
X = np.delete(X, 0, 0)
y = np.delete(y, 0)
X[:,0] = X[:,0]**(1./3.)

y[y<10] = 0
y[y>=10] = 1

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#make the ANN!
# import keras
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

# Initialising the ANN
classifier = Sequential()

# Adding the input layer 
classifier.add(Dense(4, kernel_initializer="uniform", activation = 'sigmoid', input_dim = 4 ))
# Adding the hidden layer
classifier.add(Dropout(0.2))
classifier.add(Dense(4, kernel_initializer="uniform", activation = 'sigmoid'))
classifier.add(Dropout(0.2))
classifier.add(Dense(4, kernel_initializer="uniform", activation = 'sigmoid'))
classifier.add(Dropout(0.2))
classifier.add(Dense(20, kernel_initializer="uniform", activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(20, kernel_initializer="uniform", activation = 'relu'))
classifier.add(Dropout(0.2))
# Adding the output hidden layer
classifier.add(Dense(1, kernel_initializer="uniform", activation = 'sigmoid'))
# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Fitting the ANN to the Training set 
history = classifier.fit(X_train, y_train, batch_size = 20, nb_epoch = 200)
print(history.history.keys())

classifier.save('model2_4layers_1data_bot_sigmoid.h5')

#calculate the loss and accuracy in test set
loss, accuracy = classifier.evaluate(X_test, y_test)
#predict for the test set
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)