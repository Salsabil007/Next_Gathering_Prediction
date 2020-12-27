import pandas as pd
import numpy as np
from numpy import array
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import MeanShift,KMeans, estimate_bandwidth
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.utils import np_utils
from keras.layers import Dropout
import tensorflow as tf	
#print(tf.version.VERSION)
from keras.layers import Convolution2D, MaxPooling2D

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
    labelencoder = LabelEncoder()
    data['venue_cat'] = labelencoder.fit_transform(data['venueid'])
    data['day_cat'] = labelencoder.fit_transform(data['day'])
    data['month_cat'] = labelencoder.fit_transform(data['month'])
    data['year_cat'] = labelencoder.fit_transform(data['year'])
    data = data.drop(data.columns[[1,2,9,12]], 1)
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
    data = np.delete(data, len-1, axis=0)
    '''
    yy = np.empty(1)
    for i in range(1):
        yy[i] = out '''
    if (data.shape[0] != 0):
        X.append(data)
        y = np.append(y,out)
    return X,y
def process_test(df, model, center):
    df = df.drop(df.columns[[1,4,5,6]], 1)
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
        long = temp[len-1][1]
        lat = temp[len-1][2]
        instance = np.delete(instance, len-1, axis=0)
        if (instance.shape[0] == 0):
            continue
        X = []
        X.append(instance)
        X = np.array(X)
        X = np.reshape(X,(X.shape[0],X.shape[1],X.shape[2]))
        y = np.empty(0)
        y = np.append(y,out)
        y = np_utils.to_categorical(y, n)
        '''
        _, accuracy = model.evaluate(X, y, batch_size=1, verbose=2)
        print("yesss ", accuracy)
        '''
        yhat = model.predict(X, verbose=0)
        yhat = yhat.astype('float32')
        center = center.astype('float32')
        r = tf.matmul(yhat,center) #calculating long lat from probabilities
        print(r, "long ", long, "lat ", lat)
        out1 = np.argmax(yhat[0]) #finding the index with max probability
        total += 1
        if (out1 == out):
            correct += 1
        #print("actual", out)
    print("total ", total, " correct ", correct)


def pre_padding(df,model,n):
    df = df.drop(df.columns[[1,4,5,6]], 1) #Dropping unnecessary columns
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
        y = np.empty(0)
        for j in person:
            instance = df[df.userid == j]
            instance = instance.drop(instance.columns[[0]], 1) #Dropping user id, longitude, latitude
            #print(instance.dtypes)
            instance = instance.to_numpy()
            X,y = create_batch(instance,X,y)
            #print(y.shape[0])
        X = np.array(X)
        y = np.array(y)
        X = np.reshape(X,(X.shape[0],X.shape[1],X.shape[2]))
        #print(X.shape, y.shape)
        #print(y)
        y = np_utils.to_categorical(y, n) #converting output into categorical values
        model.fit(X,y, epochs=10, batch_size=1, verbose=2)
    print("YESSSSSSSSSSSS")
    return model

#tf.compat.v1.enable_eager_execution()
#tf.compat.v1.disable_eager_execution()
data = csv_to_df()
data = convert_to_categorical(data)
#data = data.head(500)
data,n, center = clustering(data)
train, test = train_test_split(data, test_size=0.2)
#train,n = clustering(train.head(10))
model = Sequential()
model.add(LSTM(100, input_shape=(None,10)))
model.add(Dropout(0.5)) #regularization to prevent overfitting
model.add(Dense(100, activation='relu'))
model.add(Dense(n, activation='softmax')) #output layer, so output entry must be equal to number of clusters
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model = pre_padding(train,model,n)
print(n)
process_test(test, model,center)