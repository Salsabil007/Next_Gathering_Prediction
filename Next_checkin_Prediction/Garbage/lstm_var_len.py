# multivariate lstm example
import numpy as np
from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.preprocessing.sequence import pad_sequences
def pad_with(vector, pad_width, iaxis, kwargs):
	pad_value = kwargs.get('padder', '?')
	vector[:pad_width[0]]=pad_value
	vector[-pad_width[1]:]=pad_value
	return vector
sequence = [[['a','b','c'],['d','e','f']],
			[['e','f','g']],
			[['m','l','k']]]
padded =np.pad(sequence, 1, pad_with)     #pad_sequences(sequence,padding = 'post')
print(padded)
