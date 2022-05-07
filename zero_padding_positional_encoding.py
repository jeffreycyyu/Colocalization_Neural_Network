def CSV_TO_ENCODED_PADDED_INPUT(csv_pathname):
    
    #import datafram from txt file
    test_dataframe = pd.read_csv(csv_pathname, sep='\t')
    #remove heading name
    test_dataframe.columns.name = None

    #convert to float 64 numpy array
    data_frame_array = df.to_numpy(test_dataframe, dtype = 'float64')  

    #delete position column
    no_positions_array = np.delete(data_frame_array, 0, 1)


    #cuts positions to split the dataset
    #split_sample_input represents a list of variable length simulated GWAS p-values with 100 samples, 3 traits for each loci (variable number of loci per sample)
    cuts = np.sort(random.sample(range(0, data_frame_array.shape[0]-1), 100-1))

    test_input = np.split(no_positions_array , cuts, axis = 0)

    #number of samples
    n_samples = len(test_input)

    #maximum length of (maximum number of loci of) all samples
    def find_max_list(list):
        list_len = [len(i) for i in list]
        return max(list_len)
    #maximum number of loci per sample gene segment
    maximum_length = find_max_list(test_input)

    #pad input list of samples 
    padded_test_input = pad_sequences(test_input, padding='post', dtype='float64')

    #number of traits simulated
    n_traits = padded_test_input.shape[2]

    #embed positional information
    position_embedding_layer = Embedding(maximum_length, n_traits)
    position_indices = tf.range(maximum_length)
    embedded_indices = position_embedding_layer(position_indices)

    #add positional encoding to the padded test imputs
    padded_test_input += tf.cast(embedded_indices.numpy(), 'float64')
    #reassign to new name
    positional_embedded_padded_test_input = padded_test_input
    
    return positional_embedded_padded_test_input








#test import
import math
import sys
import numpy as np
import pandas as pd
import random
from pandas import DataFrame as df
import tensorflow as tf
from tensorflow.keras.layers import Embedding
from keras.preprocessing.sequence import pad_sequences

import random

random.seed(25252)

my_input = CSV_TO_ENCODED_PADDED_INPUT('/Users/jeffreyyu/Documents/Sladek/colocalization_neural_network/data/simulated_gwas.txt')
my_input
