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
from keras.layers.core import Activation
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

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
    labelencoder = LabelEncoder()
    data['cluster_grp'] = labelencoder.fit_transform(data['cluster_grp'])
    return data,n_clusters_

def str_to_date(str):
    str = str.strip() #removing any heading or tailing extra space
    x = str.split() #spliting from space #Format:Fri May 04 01:18:03 +0000 2012
    day,month,date,year = x[0],x[1],x[2],x[5]
    hour = x[3][0:2]
    return day,month,date,hour,year


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

def convert_to_categorical(data):
    labelencoder = LabelEncoder()
    data['venue_cat'] = labelencoder.fit_transform(data['venueid'])
    data['day_cat'] = labelencoder.fit_transform(data['day'])
    data['month_cat'] = labelencoder.fit_transform(data['month'])
    data['year_cat'] = labelencoder.fit_transform(data['year'])
    #train['cluster_grp'] = labelencoder.fit_transform(data['cluster_grp'])
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
feat = 0
def padding(mxlen, data, X, y):
    len = data.shape[0]
    col = data.shape[1]
    feat = col - 1
    out = data[len-1][col-1]
    data = np.delete(data, col-1, axis=1)
    dummy = np.array([0])
    for i in range(col-2):
        dummy = np.append(dummy, 0)
    p = mxlen - len
    for i in range(p):
        data = np.vstack([data,dummy])
    #print(data)
    X.append(data)
    y.append(out)
    return X,y

def pre_padding(df):
    df = df.drop(df.columns[[1,4,5,6]], 1)
    df = str_to_numeric(df)
    user = df['userid'].unique()
    #print(len(user))
    df = df.drop_duplicates()
    df.to_csv("US_nodup.csv", index = False)
    #print(len(df))
    
    #finding max instance
    maxlen = 0
    ind = -1
    X,y = [],[]
    for i in user:
        instance = df[df.userid == i]
        instance = instance.drop(instance.columns[[0]], 1)
        arr = instance.values
        length = len(instance)
        if length > maxlen:
            maxlen = length
            ind = i
    for i in user:
        instance = df[df.userid == i]
        instance = instance.drop(instance.columns[[0]], 1)
        arr = instance.values
        #arr = arr.astype('float32')
        X,y = padding(maxlen, arr, X, y)
    return np.array(X),np.array(y)
 
data = csv_to_df()
train = convert_to_categorical(data.head(50))
#train, test = train_test_split(data, test_size=0.2)
train,n_cluster = clustering(train)
#print(train.dtypes)

#print(train.dtypes)
X,y = pre_padding(train)
#print(train.dtypes)
print(X.shape[2])
#print(n_cluster)
'''
model = Sequential()
model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dense(18))
model.add(Activation('softmax'))
opt = SGD(lr=0.01, momentum=0.9)
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.fit(X, y, epochs=200, verbose=0)
print("yes")
'''
seed = 7
np.random.seed(seed)
model = Sequential()
model.add(Dense(8, input_dim=X.shape[2], activation='relu'))
model.add(Dense(n_cluster, activation='softmax'))
	# Compile model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
estimator = KerasClassifier(build_fn=model, epochs=200, batch_size=1, verbose=0)
estimator.fit(X,y)
#kfold = KFold(n_splits=3, shuffle=True, random_state=seed)
#results = cross_val_score(estimator, X, y, cv=kfold)
#print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
#model.fit(X, y, epochs=200, verbose=0)

