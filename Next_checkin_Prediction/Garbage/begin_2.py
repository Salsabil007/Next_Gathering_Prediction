import numpy as np
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf

#this one is for multi, constant length timestep regression
X,y = [],[]
a = np.array([[1,2,3],[2,3,4]])
b = np.array([[2,2,3],[3,3,4]])
c = np.array([[2,3,5],[5,8,4]])
t = np.array([[2,2,4],[3,3,6]])
testX = []
testX.append(t)
testX = np.array(testX)
testX = np.reshape(testX,(1,testX.shape[1],testX.shape[2]))
X.append(a)
X.append(b)
X.append(c)
y.append([15])
y.append([17])
y.append([27])
X = np.array(X)
y = np.array(y)
print(X.shape, y.shape)
X = np.reshape(X,(X.shape[0],X.shape[1],X.shape[2]))
model = Sequential()
model.add(LSTM(100,activation='relu',input_shape=(X.shape[1],X.shape[2]))) #only shape of timestep and feature
model.add(Dense(1))
model.compile(optimizer='adam',loss=tf.keras.losses.MeanSquaredError())
model.fit(X, y, epochs=100, batch_size=1, verbose=2)
yhat = model.predict(testX, verbose=0)
print(yhat)