import pandas as pd 
import datetime
from datetime import datetime
'''
// Converting txt file to csv
data = pd.read_csv("friendship.csv", delimiter="\t")
data.to_csv("friendship1.csv", index = False)
'''

'''
//adding header to a csv file
data = pd.read_csv("friendship1.csv",header=None)
data.to_csv("friendship2.csv",header=["userid1","userid2"],index=False)
'''

'''
// Converting time into minute
'''

'''with open('time_min.csv', 'w') as out:
    with open('check.csv', 'r') as file:
        i = -1
        for line in file:
            print(i)
            i += 1
            p = 0
            if i == 0:
                out.write(line+'\n')
                continue
            for cnt,val in enumerate(line.split(',')):
                p += 1
                if cnt !=0:
                    out.write(',')
                out.write(val)
                if p == 3:
                    s = val[26:]
                    year = int(s)
                    mon = datetime.strptime(val[4:7], '%b').month
                    day = int(val[8:10])
                    hour = int(val[11:13])
                    min = int(val[14:16])
                    time_in_minute = (year * 365 * 24 * 60 + mon * 30 * 24 * 60 + day * 24 * 60 + hour * 60 + min)
                if p == 4:
                    time_in_minute += int(val)
                    out.write(',')
                    out.write(str(time_in_minute))
'''                     

    
'''
data = pd.read_csv("check.csv")
data['time_in_minute'] = timeframe
data.to_csv("time_min.csv",index=False) 
'''
'''
timeframe = []
with open('check.csv', 'r') as file:
    i = -1
    for line in file:
        #print(i)
        i += 1
        p = 0
        if i == 0:
            #out.write(line+'\n')
            continue
        if i == 17720249:
            timeframe.append(9999)
            continue
        for cnt,val in enumerate(line.split(',')):
            p += 1
            if cnt !=0:
                out.write(',')
            out.write(val)
            if p == 3:
                try:
                    s = val[26:]
                    year = int(s)
                    mon = datetime.strptime(val[4:7], '%b').month
                    day = int(val[8:10])
                    hour = int(val[11:13])
                    min = int(val[14:16])
                    time_in_minute = (year * 365 * 24 * 60 + mon * 30 * 24 * 60 + day * 24 * 60 + hour * 60 + min)
                except:
                    print(i)
                    print(val+"***")
            if p == 4:
                time_in_minute += int(val)
                timeframe.append(time_in_minute)
                #out.write(',')
                #out.write(str(time_in_minute))


chunks = pd.read_csv("check.csv", chunksize=100000)
data = pd.concat(chunks)
data['time_in_minute'] = timeframe

data.to_csv("check2.csv", index = False)
'''