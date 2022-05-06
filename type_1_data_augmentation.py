import math
import sys
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
import random
import matplotlib.pyplot as plt
from pandas import DataFrame as df
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Input, Embedding, Activation, Add, Conv1D, Conv1DTranspose, LSTM, Layer, LayerNormalization, ReLU, Embedding, Bidirectional
from keras.preprocessing.sequence import pad_sequences
import tensorflow_addons as tfa
from tensorflow.keras.models import Model, Sequential
from keras.utils.vis_utils import plot_model
import edward2 as ed
import warnings

#type 1 data augmentation (set lowest p-value equal to second lowest (i.e., set to be tied for 1st)

from ipynb.fs.full.test_data import CSV_TO_ENCODED_PADDED_INPUT

random.seed(25252)

csv_pathname = '/Users/jeffreyyu/Documents/Sladek/colocalization_neural_network/simulated_gwas.txt'

my_input = CSV_TO_ENCODED_PADDED_INPUT(csv_pathname)


test_dataframe = pd.read_csv(csv_pathname, sep='\t')

test_dataframe.columns.name = None

print(test_dataframe)

#example plot of first 1000 positions
%matplotlib qt
plt.scatter(test_dataframe['Position'][0:1000], -1*np.log10(test_dataframe['trait_1'][0:1000]), s=1)
# plt.scatter(test_dataframe['Position'][0:1000], -1*np.log10(test_dataframe['trait_2'][0:1000]), s=1)
# plt.scatter(test_dataframe['Position'][0:1000], -1*np.log10(test_dataframe['trait_3'][0:1000]), s=1)
plt.show()
plt.close()


#====data augmentation

#dataframe of p values on the -log10 scale for 40 positions
df = pd.DataFrame(-1*np.log10(test_dataframe['trait_1'][0:40]))
#df = pd.DataFrame(test_dataframe['trait_1'][0:999])


# number of points to be checked upstream and downstream
n = 10  

#find local maxima/minima
df['min'] = df.iloc[argrelextrema(df.trait_1.values, np.less_equal,
                    order=n)[0]]['trait_1']
df['max'] = df.iloc[argrelextrema(df.trait_1.values, np.greater_equal,
                    order=n)[0]]['trait_1']


# Plot results
plt.scatter(df.index, df['min'], c='r')
plt.scatter(df.index, df['max'], c='g')
plt.plot(df.index, df['trait_1'])
plt.show()
#plt.close()

#this is a dataframe with 3 columns representing: trait_1's -log10(p-value); minimum value (either NaN or smae as in trait_1); maximum value (either NaN or smae as in trait_1)
print(df)

#======

#this is how upstream or downstream a local maxima/minima considers it's 'neighborhood'
span_length = 5
#this is the row index of all maxima
maxima_indices = np.where(~np.isnan(df['max']))[0]

#============type 1


#for every local maxima of trait_1's -log10(p_value), set the local maxima to be equal to the second highest point within the 'neighborhood (i.e., now maxima is 'tied' for first)
for i in range(0, len(maxima_indices)):
    df['trait_1'][maxima_indices[i]] = sorted(df['trait_1']
                                              [maxima_indices[i] - span_length:maxima_indices[i] + span_length],
                                              reverse = True)[1]


#same plot but now local maxima should be equal to second highest -log10(p_value) within the region
plt.scatter(df.index, df['min'], c='r')
plt.scatter(df.index, df['max'], c='g')
plt.plot(df.index, df['trait_1'])
plt.show()
#plt.close()

#same dataframe but now local maxima should be equal to second highest -log10(p_value) within the region (note: do not use anything from 'max' and 'min' columns)
print(df)
