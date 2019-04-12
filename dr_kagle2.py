# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 19:44:15 2019

@author: Gaurav Mehra
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 12:37:32 2019

@author: Gaurav Mehra
"""

# CNN to recognize handwritten digits
# Test set accuarcy: 98.93%


import numpy
import pandas as pd
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

# fix random seed
seed = 7
numpy.random.seed(seed)


# load data
dataset1 = pd.read_csv('train.csv')
dataset2 = pd.read_csv('test.csv')
X = dataset1.iloc[:,1:785].values
y = dataset1.iloc[:, 0].values


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
# normalize
X_train = X_train/255
X_test = X_test/255

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
num_pixels = 784

# define the larger model
def larger_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(1,28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(150, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
	# Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return model

# model build
model = larger_model()
# fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs = 12, batch_size = 150, verbose = 1)
# model evaluation
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

# predict
X_k = dataset2.iloc[:,:].values
# reshape to be [samples][pixels][width][height]
X_k = X_k.reshape(X_k.shape[0], 1, 28, 28).astype('float32')
y_res = model.predict_classes(X_k, verbose 1)
from numpy import argmax
y_res = argmax(y_res, axis = 1)
y_res = y_res.reshape((28000,1))


import numpy
numpy.savetxt("res.csv", y_res, fmt= '%i', delimiter=",")

