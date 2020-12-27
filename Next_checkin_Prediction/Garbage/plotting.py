import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Dataset_US.csv")
#userids = [19, 1668, 854, 5526]
userids = [19]
for userid in userids:
    data = data[data.userid == userid].sort_values('time_in_minute')
    plt.subplot(2,1,1)
    plt.plot(data['time_in_minute'],data['latitude'], '-o')
    #plt.plot(data1['time_in_minute'],data1['latitude'], 'x')
    plt.xlabel('time')
    plt.ylabel('latitude')
    plt.subplot(2,1,2)
    plt.plot(data['time_in_minute'],data['longitude'], '-o')
    plt.xlabel('time')
    plt.ylabel('longitude')
    #plt.plot(data1['time_in_minute'],data1['longitude'], 'x')
    print(userid)

plt.tight_layout()
plt.show()