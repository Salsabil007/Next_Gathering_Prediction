import numpy as np
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf

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
testX = np.array([[10,20,30]])
testX = np.reshape(testX,(1,X.shape[1],1))

X = np.reshape(X,(X.shape[0],X.shape[1],1))
model = Sequential()
model.add(LSTM(100,activation='relu',input_shape=(X.shape[1],X.shape[2]))) #only shape of timestep and feature
#model.add(Dense(1))
model.compile(optimizer='adam',loss=tf.keras.losses.MeanSquaredError())

model.fit(X, y, epochs=10, batch_size=1, verbose=2)

yhat = model.predict(testX, verbose=0)
print(yhat," yes")