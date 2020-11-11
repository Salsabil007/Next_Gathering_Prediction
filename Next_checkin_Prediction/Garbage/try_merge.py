import pandas as pd 


data1 = pd.read_csv("data_JP.csv")
data1 = data1.drop(data1.columns[[3,5]], 1)
#data1 = data1.head(100)
data2 = pd.read_csv("data_JP2.csv")
data2 = data2.drop(data2.columns[[3,5]], 1)
#data2 = data2.head(100)
result = pd.merge(data1,data2,how='inner',on='venueId')
result = result[result.userId != result.userId2]
result.to_csv("merge_JP.csv",index = False)
