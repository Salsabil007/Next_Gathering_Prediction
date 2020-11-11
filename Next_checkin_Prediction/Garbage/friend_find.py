import pandas as pd 

'''
//removing quotation of every data
data = pd.read_csv("athena_US_sorted.csv",quotechar='"',skipinitialspace=True)
data.to_csv("test_US.csv", index = False)
print(data.dtypes)
'''
'''
data = pd.read_csv("friendship2.csv")
print(data.dtypes)
'''
'''
is_friend = []
data = pd.read_csv("friendship2.csv")
with open('test_US.csv', 'r') as file:
        i = -1
        for line in file:
            i += 1
            if i == 0:
                continue
            p = 0
            for cnt,val in enumerate(line.split(',')):
                p += 1
                if p == 1:
                    id1 = int(val)
                elif p == 5:
                    id2 = int(val)
                    data1 = data[(data.userid1 == id1) & (data.userid2 == id2)]
                    data2 = data[(data.userid1 == id2) & (data.userid2 == id1)]
                    if len(data1.index) == 0 and len(data2.index) == 0:
                        is_friend.append("NO")
                    else:
                        is_friend.append("YES")

                else:
                    continue

data = pd.read_csv("test_US.csv")
data['is_friend'] = is_friend
data.to_csv("final_US.csv", index = False)         
'''      
'''
data = pd.read_csv("final_US.csv")
data = data[data.is_friend == "YES"]
data.to_csv("US_data_with_friend.csv", index = False)   
'''
'''
data = pd.read_csv("POI.csv")
data = data[data.country_code == "US"]
data.to_csv("POI_US.csv", index = False)
'''

'''
//adding venue data to each entry
data2 = pd.read_csv("POI_US.csv")
data1 = pd.read_csv("US_data_with_friend.csv")
result = pd.merge(data1, data2, how = "inner", on = "venueid")
result.to_csv("Dataset_US.csv", index = False)
'''

data1 = pd.read_csv("Dataset_US.csv")
data1 = data1.sort_values(['userid','time_in_minute'])
data1.to_csv("Dataset_US.csv", index = False)