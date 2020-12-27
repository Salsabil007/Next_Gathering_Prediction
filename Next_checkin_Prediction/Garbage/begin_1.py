import numpy as np
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
#import tensorflow as tf
#this one is for single, constant length timestep regression
#process input
X,y = [],[]
X.append([1,2,3])
X.append([5,6,7])
X.append([3,4,5])
#p = np.array([6])
y.append([6])
y.append([18])
y.append([12])
X = np.array(X)
y = np.array(y)
print(X.shape, y.shape)

X = np.reshape(X,(X.shape[0],X.shape[1],1))
#print(X.shape)

testX = np.array([[10,20,30],[7,8,9],[10,1,12]])
testX = np.reshape(testX,(3,X.shape[1],1))
testy = np.array([[60],[24],[23]])
print(testX.shape, testy.shape)

model = Sequential()
model.add(LSTM(100,activation='relu',input_shape=(X.shape[1],X.shape[2]))) #only shape of timestep and feature
model.add(Dense(1))
#model.compile(optimizer='adam',loss=tf.keras.losses.MeanSquaredError())
model.compile(optimizer='adam',loss='mean_squared_error')

model.fit(X, y, epochs=10, batch_size=1, verbose=2)

#yhat = model.predict(testX, verbose=0)
print("yes")

score = model.evaluate(testX, testy, verbose=0)
print(score)
