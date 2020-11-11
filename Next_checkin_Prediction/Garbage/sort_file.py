import pandas as pd 

'''
chunks = pd.read_csv("check3.csv", chunksize=100000)
data = pd.concat(chunks)
data = data.sort_values(by=['venueId'])
data.to_csv("data_sort_venue.csv", index = False)
'''
'''
data = pd.read_csv("raw_POIs.txt", delimiter = "\t")
data.to_csv("POI.csv", header = ["venueId","latitude","longitude", "venue catagory", "country code"] , index = False)
'''
'''
chunks = pd.read_csv("POI.csv", chunksize=100000)
data = pd.concat(chunks)
data = data.sort_values(by=['country code'])
data.to_csv("POI_sort_venue.csv", index = False)
'''
'''
chunks = pd.read_csv("POI_sort_venue.csv", chunksize=100000)
data = pd.concat(chunks)
print(data['country code'].unique())
'''
'''
dataframe = []
my_dict = {}
str = ""
with open('POI_sort_venue.csv', 'r') as file:
    i = -1
    for line in file:
        i += 1
        p = 0
        if i == 0:
            continue
        for cnt,val in enumerate(line.split(',')):
            p += 1
            if p == 1:
                str = val
            if p == 5:
                my_dict[str] = val[0:2]



with open('data_sort_venue.csv', 'r') as file:
    i = -1
    for line in file:
        i += 1
        p = 0
        if i == 0:
            continue
        for cnt,val in enumerate(line.split(',')):
            p += 1
            if p == 2:
                str = val
            if p == 5:
                dataframe.append(my_dict[str])

chunks = pd.read_csv("data_sort_venue.csv", chunksize=100000)
data = pd.concat(chunks)
data['country_code'] = dataframe
data.to_csv("data_with_CC_sorted.csv", index = False)

'''
'''
chunks = pd.read_csv("data_with_CC_sorted.csv", chunksize=100000)
data = pd.concat(chunks)
data = data[data.country_code == 'JP']     
data.to_csv("data_JP.csv", index = False)    
'''
'''
chunks = pd.read_csv("data_US.csv", chunksize=100000)
data = pd.concat(chunks)
data = data.sort_values(by=['userId'])
data.to_csv("US_sort.csv", index = False)
'''
'''
chunks = pd.read_csv("data_US.csv", chunksize=100000)
data1 = pd.concat(chunks)
data1 = data1.drop(data1.columns[[3,5]], 1)
data1.to_csv("data_USS.csv", index = False)
chunks = pd.read_csv("data_US2.csv", chunksize=100000)
data2 = pd.concat(chunks)
data2 = data2.drop(data2.columns[[3,5]], 1)
data2.to_csv("data_USS2.csv", index = False)
'''

chunks = pd.read_csv("from_athena_US.csv", chunksize=100000)
data1 = pd.concat(chunks)
data1 = data1.sort_values(by=['userid','userid2'])
data1.to_csv("athena_US_sorted.csv", index = False)
                























