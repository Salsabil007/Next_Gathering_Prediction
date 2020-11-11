import pandas as pd
'''
chunks = pd.read_csv("check2.csv", chunksize=100000)
data = pd.concat(chunks)
data = data.sort_values(by=['time_in_minute'])
data.to_csv("check3.csv", index = False)
'''

chunks = pd.read_csv("check3.csv", chunksize=100000)
data1 = pd.concat(chunks)
data1.drop(['UTC time', 'Timezone offset in min'], axis=1)
chunks = pd.read_csv("check4.csv", chunksize=100000)
data2 = pd.concat(chunks)
data2.drop(['UTC time2', 'Timezone offset in min2'], axis=1)
result = pd.merge(data1, data2, how = 'inner', on = 'venueId')
result = result[result.userId != result.userId2]
result.to_csv("merge.csv",index = False)


