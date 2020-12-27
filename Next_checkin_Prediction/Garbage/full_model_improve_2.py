import pandas as pd
import numpy as np
from numpy import array
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import MeanShift,KMeans, estimate_bandwidth
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.layers.core import Activation
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.utils import np_utils
from keras.layers import Dropout
from keras import backend as K
from keras.layers import Convolution2D, MaxPooling2D
from math import radians, cos, sin, asin, sqrt
from sklearn.metrics import accuracy_score

import tensorflow as tf
import matplotlib.pyplot as plt
import math
#print(tf.version.VERSION)

### converted output from cluster to long lat

def distance(pos1, lat2, lon2):
    lat1 = pos1[:, 0]
    lon1 = pos1[:, 1]
    #lat1, lon1 = origin
    #lat2, lon2 = destination
    radius = 6371 # km

    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c

    return d
def tf_atan2(y, x):
    angle = tf.where(tf.greater(x,0.0), tf.atan(y/x), tf.zeros_like(x))
    angle = tf.where(tf.logical_and(tf.less(x,0.0),  tf.greater_equal(y,0.0)), tf.atan(y/x) + np.pi, angle)
    angle = tf.where(tf.logical_and(tf.less(x,0.0),  tf.less(y,0.0)), tf.atan(y/x) - np.pi, angle)
    angle = tf.where(tf.logical_and(tf.equal(x,0.0), tf.greater(y,0.0)), 0.5*np.pi * tf.ones_like(x), angle)
    angle = tf.where(tf.logical_and(tf.equal(x,0.0), tf.less(y,0.0)), -0.5*np.pi * tf.ones_like(x), angle)
    angle = tf.where(tf.logical_and(tf.equal(x,0.0), tf.equal(y,0.0)), np.nan * tf.zeros_like(x), angle)
    return angle


def haversine(pos1, pos2):
    #print("pos1 ", pos1, " pos2 ", pos2)
    '''
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    '''
    # convert decimal degrees to radians 
    lat1 = pos1[:, 0]
    lon1 = pos1[:, 1]
    lat2 = pos2[:, 0]
    lon2 = pos2[:, 1]
    #lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = tf.abs(lon1 - lon2) * np.pi / 180
    dlat = tf.abs(lat1 - lat2) * np.pi / 180
    a = tf.sin(dlat/2)**2 + tf.cos(lat1 * np.pi / 180) * tf.cos(lat2 * np.pi / 180) * tf.sin(dlon/2)**2
    c = 2 * tf_atan2(tf.sqrt(a), tf.sqrt(1 - a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def csv_to_df():
    day = []
    month = []
    date = []
    hour = []
    year = []
    holiday = [] # sat,sun - 1, fri - 0, rest - 2
    with open('Dataset_US.csv', 'r') as file:
        i = -1
        for line in file:
            i += 1
            if i == 0:
                continue
            p = 0
            for cnt,val in enumerate(line.split(',')):
                p += 1
                if p == 3:
                    d,m,dt,h,y = str_to_date(val)
                    day.append(d)
                    month.append(m)
                    date.append(dt)
                    hour.append(h)
                    year.append(y)
                    if d == 'Sat' or d == 'Sun':
                        holiday.append("1")
                    elif d == 'Fri':
                        holiday.append("0")
                    else:
                        holiday.append("2")
                    break
    dataframe = pd.read_csv("Dataset_US.csv")
    dataframe = dataframe.drop(dataframe.columns[[3,5,6,7]], 1)
    dataframe['day'] = day
    dataframe['month'] = month
    dataframe['date'] = date
    dataframe['hour'] = hour
    dataframe['year'] = year
    dataframe['holiday'] = holiday
    return dataframe
def clustering(data):
    #coordinate = data.as_matrix(columns = ['latitude','longitude'])
    coordinate = data[['latitude','longitude']].to_numpy()
    bandwidth = estimate_bandwidth(coordinate, quantile = 0.2)
    meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    meanshift.fit(coordinate)
    labels = meanshift.labels_
    cluster_centers = meanshift.cluster_centers_
    n_clusters_ = len(np.unique(labels))
    data['cluster_grp'] = np.nan
    for i in range(len(coordinate)):
        data['cluster_grp'].iloc[i] = labels[i]
    return data,n_clusters_, cluster_centers

def str_to_date(str):
    str = str.strip() #removing any heading or tailing extra space
    x = str.split() #spliting from space #Format:Fri May 04 01:18:03 +0000 2012
    day,month,date,year = x[0],x[1],x[2],x[5]
    hour = x[3][0:2]
    return day,month,date,hour,year
def convert_to_categorical(data):
    #print("gooo ", data.dtypes)
    labelencoder = LabelEncoder()
    data['venue_cat'] = labelencoder.fit_transform(data['venueid'])
    data['day_cat'] = labelencoder.fit_transform(data['day'])
    data['month_cat'] = labelencoder.fit_transform(data['month'])
    data['year_cat'] = labelencoder.fit_transform(data['year'])
    data['venue_type'] = labelencoder.fit_transform(data['venue_catagory'])
    data = data.drop(data.columns[[1,2,9,12,6]], 1)
    #print("nooo ", data.dtypes)
    return data
def str_to_numeric(df):
    df['userid'] = pd.to_numeric(df['userid'])
    #df['userid2'] = pd.to_numeric(df['userid2'])
    df['latitude'] = pd.to_numeric(df['latitude'])
    df['longitude'] = pd.to_numeric(df['longitude'])
    df['date'] = pd.to_numeric(df['date'])
    df['hour'] = pd.to_numeric(df['hour'])
    df['holiday'] = pd.to_numeric(df['holiday'])
    df['venue_cat'] = pd.to_numeric(df['venue_cat'])
    df['day_cat'] = pd.to_numeric(df['day_cat'])
    df['month_cat'] = pd.to_numeric(df['month_cat'])
    df['year_cat'] = pd.to_numeric(df['year_cat'])
    df['cluster_grp'] = pd.to_numeric(df['cluster_grp'])
    return df
def create_batch(data, X, y):
    len = data.shape[0]
    col = data.shape[1]
    #print("col",col)
    out = data[len-1][col-1]
    a,b = np.empty(2), np.empty(1)
    #for i in range(1):
    a[0] = data[len-1][1]
    a[1] = data[len-1][2]
    data = np.delete(data, len-1, axis=0)
    
    if (data.shape[0] != 0):
        X.append(data)
        #y = np.append(y,a)
        y.append(a)
    #print("X ", X, " y ", y)
    return X,y
def process_test(df, model, center):
    df = df.drop(df.columns[[1,4,5]], 1)
    df = str_to_numeric(df)
    df = df.drop_duplicates()
    person = df['userid'].unique()
    #person = person.to_numpy()
    #person = person.flatten()
    total = 0
    correct = 0
    for i in person:
        instance = df[df.userid == i]
        temp = instance
        temp = temp.to_numpy()
        instance = instance.drop(instance.columns[[0]], 1)
        instance = instance.to_numpy()
        len = instance.shape[0]
        col = instance.shape[1]
        out = instance[len-1][col-1]
        long = temp[len-1][2]
        lat = temp[len-1][1]
        a = np.empty(2)
    #for i in range(1):
        a[0] = instance[len-1][1]
        a[1] = instance[len-1][2]
        instance = np.delete(instance, len-1, axis=0)
        if (instance.shape[0] == 0):
            continue
        X,y = [],[]
        X.append(instance)
        y.append(a)
        X = np.array(X)
        y = np.array(y)
        X = np.reshape(X,(X.shape[0],X.shape[1],X.shape[2]))
        #y = np.empty(0)
        #y = np.append(y,out)
        #y = np_utils.to_categorical(y, n)
        '''
        _, accuracy = model.evaluate(X, y, batch_size=1, verbose=2)
        print("yesss ", accuracy)
        '''
        yhat = model.predict(X, verbose=0)
        print("predic ",yhat, "actual ", lat, " ", long)
        loss = distance(yhat, lat,long)
        print("loss ", loss)
        plt.plot(hist.history["loss"])
        plt.title("model loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")

    plt.show()
        

def pre_padding(df,model,n):
    #print("before ", df.dtypes)
    df = df.drop(df.columns[[1,4,5]], 1) #Dropping unnecessary columns
    #print("now ", df.dtypes)
    df = str_to_numeric(df) #Converting the values into numeric
    df = df.drop_duplicates() #Dropping duplicate rows
    df.to_csv("US_nodup.csv", index = False)
    value_counts = df['userid'].value_counts() #counting number of rows for each distinct userid

    #storing each id and corresponding counts of rows in a file
    counts = pd.DataFrame(value_counts)
    counts = counts.reset_index()
    counts.columns = ['unique_id', 'counts']
    counts.to_csv("count.csv", index = False)
    counts = pd.read_csv("count.csv")

    x = counts['counts'].unique() #finding unique counts. user with unique count have same length of timestep and will be in same batch during training
    for i in x:
        if (i==1):
            continue
        person = counts[counts.counts == i]
        person = person.drop(person.columns[[1]], 1)
        person = person.to_numpy()
        len = person.shape[0]
        person = person.flatten() #person was column. so converting it into a row
        #print(person)
        X,y = [],[]
        #y = np.empty(0)
        for j in person:
            instance = df[df.userid == j]
            instance = instance.drop(instance.columns[[0]], 1) #Dropping user id, longitude, latitude
            #print("batch ", instance.dtypes)
            instance = instance.to_numpy()
            X,y = create_batch(instance,X,y)
            #print(y.shape[0])
        X = np.array(X)
        y = np.array(y)
        X = np.reshape(X,(X.shape[0],X.shape[1],X.shape[2]))
        #y = np.reshape(y,(y.shape[0],y.shape[1]))
        #print(X.shape, y.shape)
        #print(y)
        #y = np_utils.to_categorical(y, n) #converting output into categorical values
        hist = model.fit(X,y, epochs=5, batch_size=1, verbose=2)
    #print("YESSSSSSSSSSSS")
    return model, hist

#tf.compat.v1.enable_eager_execution()
#tf.compat.v1.disable_eager_execution()
data = csv_to_df()

data = convert_to_categorical(data)
#print(data.dtypes)
data = data.head(500)
data,n, center = clustering(data)

train, test = train_test_split(data, test_size=0.2)
#train,n = clustering(train.head(10))
model = Sequential()
model.add(LSTM(100, input_shape=(None,11)))
model.add(Dropout(0.5)) #regularization to prevent overfitting
model.add(Dense(100, activation='relu'))
model.add(Dense(n))
model.add(Activation('softmax'))
#model.add(Dense(n, activation='softmax')) #output layer, so output entry must be equal to number of clusters
center = center.astype('float32')
def linearlayer(softmaxout):
    #print("yesss ",softmaxout)
    #print(softmaxout.shape, " ", center.shape)
    return tf.matmul(softmaxout,center)
#model.add(Dense(1, activation =linearlayer))
model.add(Activation(linearlayer))
model.compile(loss=haversine, optimizer ='adam', metrics=['accuracy'])
model,hist = pre_padding(train,model,n)
'''
plt.plot(hist.history["loss"])
#plt.plot(hist.history["val_loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "val"], loc="upper left")
plt.show()
'''
process_test(test, model,center)