import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
#import pandas.rpy.common as com

'''
dataset = pandas.read_csv('Dataset_US.csv', usecols=[9], engine='python')
plt.plot(dataset)
plt.show()
'''
data = pd.read_csv("Dataset_US.csv")
#print(data.info())
#print(data.describe())
#print(data.latitude.unique())
#print(data.userid.value_counts())

sns.set_theme(style="white")
corr = data.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
print("yes")