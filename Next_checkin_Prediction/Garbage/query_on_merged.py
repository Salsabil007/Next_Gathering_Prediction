import pandas as pd 

'''
// for calculating time difference between each pair
time_diff = []
with open('merge_AD.csv', 'r') as file:
        i = -1
        for line in file:
            i += 1
            if i % 1000000 != 0:
                print(i)
            if i == 0:
                continue
            p = 0
            for cnt,val in enumerate(line.split(',')):
                p += 1
                if p == 4:
                    t1 = int(val)
                elif p== 7:
                    t2 = int(val)
            time_diff.append(abs(t1-t2))
data1 = pd.read_csv("merge_AD.csv")
data1['time_diff'] = time_diff
data1.to_csv("merge_AD_with_timediff.csv", index = False)
'''

data1 = pd.read_csv("merge_AD_with_timediff.csv")
#print(len(data1))
data1 = data1[data1.time_diff <= 120]
#print(len(data1))
data1.to_csv("merge2_AD.csv", index = False)