import sys
import pandas as pd
import random
import numpy as np

#import datafram from txt file
test_dataframe = pd.read_csv('/Users/jeffreyyu/Documents/Sladek/colocalization_neural_network/simulated_gwas.txt', sep='\t')
#remove heading name
test_dataframe.columns.name = None
#sanity: dimensions of df
print(test_dataframe.shape)

#convert to float 64 numpy array
data_frame_array = df.to_numpy(test_dataframe, dtype = 'float64')
#sanity: dimensions and show
print(data_frame_array.shape)     
#data_frame_array

#delete position column
no_positions_array = np.delete(data_frame_array, 0, 1)
print(no_positions_array.shape)
#no_positions_array


#cuts positions to split the dataset
cuts = np.sort(random.sample(range(0, data_frame_array.shape[0]-1), 100-1))
print(len(cuts))
#cuts
split_sample_input = np.split(no_positions_array , cuts, axis = 0)
print(split_sample_input[0].shape)
print(split_sample_input[1].shape)
print(split_sample_input[2].shape)

#split_sample_input represents a list of variable length simulated GWAS p-values with 100 samples, 3 traits for each loci (variable number of loci per sample)

