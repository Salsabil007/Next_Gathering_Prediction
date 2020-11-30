import numpy as np
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from keras.utils import np_utils
from keras.layers import Dropout
#this one is for single, constant length timestep classification
#important note: categorical_crossentropy loss function is working fine. but sparse_categorical_crossentropy is giving error!!
#process input
X,y = [],[]
X.append([1,2,3])
X.append([5,6,7])
X.append([3,4,5])
#p = np.array([6])
y.append([0])
y.append([1])
y.append([1])
X = np.array(X)
y = np.array(y)
X = np.reshape(X,(X.shape[0],X.shape[1],1))
y = np_utils.to_categorical(y, 2)
print(X.shape,y.shape)
model = Sequential()
model.add(LSTM(100, input_shape=(X.shape[1],1)))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=10, batch_size=1, verbose=2)
testX = np.array([[10,20,30]])
testX = np.reshape(testX,(1,X.shape[1],1))
yhat = model.predict(testX, verbose=0)
print(yhat)

'''
model = Sequential()

model.add(Dense(500,activation = 'relu', input_shape = (1,X.shape[1])))
model.add(Dense(2, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
              
model.fit(X, y, epochs=10, batch_size=1, verbose=2)
'''
'''
testX = np.array([[10,20,30],[7,8,9],[10,1,12]])
testX = np.reshape(testX,(3,X.shape[1],1))
testy = np.array([[60],[24],[23]])
print(testX.shape, testy.shape)
model = Sequential()
model.add(LSTM(100,activation='relu',input_shape=(X.shape[1],X.shape[2]))) #only shape of timestep and feature
model.add(Dense(1))
model.compile(optimizer='adam',loss=tf.keras.losses.MeanSquaredError())
model.fit(X, y, epochs=100, batch_size=1, verbose=2)
yhat = model.predict(testX, verbose=0)
'''

