class AUGMENT_DATA:
    """
    Data augmentation.
    """ 
    def __init__(
        self,
        augmentation_type: int,
        extremum_window: int,
        neighborhood_window: int,
        name: str = 'augment_data'
        ):
        """
        Argument(s):
            augmentation_type: integer denoting type of data augmentation to preform; type 1 sets the local minima equal to the second lowest value in the neighborhood; type n for n > 1 switches p-values of the local minima with the nth lowest p-value in the neighborhood 
            extremum_window: integer denoting how far to look upsteam/downstream when evaluating whether a loci is a local minima
            neighborhood_window: integer denoting how far to look for a replacement value when local minima is augmented
            name: name
        """
        print('AUGMENT_DATA.__init__')
        
        self.augmentation_type = augmentation_type
        self.extremum_window = extremum_window
        self.neighborhood_window = neighborhood_window  
        self._name = name
        
    def __call__(self, input_sequences):
        """
        Argument(s):
            input_sequence: 1-D input sequence of p-values sorted by position
        Returns:
            self.output_sequence: augmented input sequence
        """
        print('AUGMENT_DATA.__call__')
        
        #initialize empty dataframe
        dataframe = pd.DataFrame()
        
        #append origional p-values to dataframe
        dataframe['trait'] = input_sequences
        
        #find all local minima and report p-values; value is NaN if loci is not a local minima
        dataframe['minimum'] = dataframe.iloc[argrelextrema(dataframe.trait.values,
                                                     np.less_equal,
                                                     order=self.extremum_window)[0]]['trait']
        
        #row indices of all local minima
        minima_indices = np.where(~np.isnan(dataframe['minimum']))[0]
        
        #distinguish between type 1 and type n (where n>1) data augmentations
        if self.augmentation_type == 1:
            
            #set all local minima in the sequence equal to second lowest p-value in the neighborhood
            for i in range(0, len(minima_indices)):
                #replace minima as
                dataframe['trait'][minima_indices[i]] = sorted(dataframe['trait']
                                                        [minima_indices[i] - self.neighborhood_window:
                                                         minima_indices[i] + self.neighborhood_window],
                                                               reverse = True)[1]
            
            #return augmented sequence
            return dataframe['trait']
            
        #distinguish between type 1 and type n (where n>1) data augmentations
        elif self.augmentation_type > 1:
            
            #store origional trait p-values for importing later
            trait_storage = list(dataframe['trait'])
            
            #switch p-values for local minima in the sequence with the nth lowest p-value
            for i in range(0, len(minima_indices)):
                
                #local minima to be switched
                local_minima = dataframe['trait'][minima_indices[i]]
                
                #nth lowest p-value in the neighborhood
                nth_minima = sorted(dataframe['trait'][minima_indices[i] - self.neighborhood_window:
                                                                 minima_indices[i] + self.neighborhood_window],
                                    reverse = True)[self.augmentation_type - 1]
                
                #set local minima to nth lowest p-value in the neighborhood
                dataframe['trait'][minima_indices[i]] = nth_minima
                
                #set nth lowest p-value in the neighborhood to local minima
                dataframe['trait'][trait_storage.index(nth_minima)] = local_minima
                
                #return augmented sequence
                return dataframe['trait']
        
        #error if integer is negative
        else: raise ValueError('augmentation_type was not a positive integer')
    

    
    
    
    
    
    

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


#this will be the input to the function
print(test_dataframe)


#example plot of first 1000 positions
%matplotlib qt
plt.scatter(test_dataframe['Position'][0:1000], -1*np.log10(test_dataframe['trait_1'][0:1000]), s=1)
# plt.scatter(test_dataframe['Position'][0:1000], -1*np.log10(test_dataframe['trait_2'][0:1000]), s=1)
# plt.scatter(test_dataframe['Position'][0:1000], -1*np.log10(test_dataframe['trait_3'][0:1000]), s=1)
plt.show()
#plt.close()


#====data augmentation TEST



augmentation_test = AUGMENT_DATA(1, 10, 5)

augmented_data = augmentation_test(test_dataframe['trait_1'])

print('origional')
test_dataframe['trait_1']

print('test')
augmented_data
    
    
    

    
    
