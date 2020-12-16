import numpy as np
import pandas as pd
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from keras.utils import np_utils
from keras.layers import Dropout
#this one is for multiple, constant length timestep classification
#important note: categorical_crossentropy loss function is working fine. but sparse_categorical_crossentropy is giving error!!
#process input
a = pd.DataFrame(data={'A': [1, 2], 'B': [4, 5], 'C': [7, 8]})
a = a.to_numpy()
b = pd.DataFrame(data={'A': [3, 2], 'B': [4, 56], 'C': [7, 8]})
b = b.to_numpy()
c = pd.DataFrame(data={'A': [4, 2], 'B': [4, 5], 'C': [6, 8]})
c = c.to_numpy()

X,y = [],[]
t = np.array([[2,2,4],[3,3,6]])
testX = []
testX.append(t)
testX = np.array(testX)
testX = np.reshape(testX,(1,testX.shape[1],testX.shape[2]))
X.append(a)
X.append(b)
X.append(c)
y = np.empty(3)
for i in range(3):
    y[i] = 1
'''
y.append([0])
y.append([1])
y.append([1])
'''
X = np.array(X)
y = np.array(y)
X = np.reshape(X,(X.shape[0],X.shape[1],X.shape[2]))
y = np_utils.to_categorical(y, 2)
print(X.shape,y.shape)

model = Sequential()
model.add(LSTM(100, input_shape=(X.shape[1],X.shape[2])))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=10, batch_size=1, verbose=2)
yhat = model.predict(testX, verbose=0)
print(yhat)

