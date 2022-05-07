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


#MODEL INPUT

#set path to 'data' folder
sys.path.insert(0, '/Users/jeffreyyu/Documents/Sladek/colocalization_neural_network/data')

#import function to read, zero pad, and positionally encode a tab delimited .txt data file
from ipynb.fs.full.zero_padding_positional_encoding import IMPORT_ZERO_PAD_POSITIONAL_ENCODE

#set random seed
random.seed(25252)

#data file pathname
pathname = '/Users/jeffreyyu/Documents/Sladek/colocalization_neural_network/data/simulated_gwas.txt'

#read, zero pad, and positionally encode
test_input = IMPORT_ZERO_PAD_POSITIONAL_ENCODE(pathname)
#input shape
input_shape = test_input.shape


#print input shape and input
print(test_input.shape)
print(test_input)




#create a dataframe version of the input for viewing
test_dataframe = pd.read_csv(pathname, sep='\t')
test_dataframe.columns.name = None
print(test_dataframe)
#open plots in a seperate window for matplotlibx
%matplotlib qt
#example plot of first 1000 positions for all 3 traits with p-values on the -log10 scale (like in a manhattan plot)
plt.scatter(test_dataframe['Position'][0:1000], -1*np.log10(test_dataframe['trait_1'][0:1000]), s=1)
plt.scatter(test_dataframe['Position'][0:1000], -1*np.log10(test_dataframe['trait_2'][0:1000]), s=1)
plt.scatter(test_dataframe['Position'][0:1000], -1*np.log10(test_dataframe['trait_3'][0:1000]), s=1)
plt.show()


# #Old simulated input (do not use)
# #Model Input
# #generate data 400 gene segments each of length 100bp/loci and with 5 traits/cell-types
# #test: GWAS, eQTL1, eQTL2, eQTL3, eQTL4 p-values for 100 bp's (100 loci/positions)
# #randomly generate 5 vectors which represent the 5 trait's p-values for 100 loci each
# input_shape = (400, 100, 5)
# test_input = np.random.random(input_shape)







#MULTI_HEAD_ATTENTION
from ipynb.fs.full.multi_head_attention import MULTI_HEAD_ATTENTION

#pre-specify multi-head attention layer parameters
multihead_layer_test = MULTI_HEAD_ATTENTION(
        n_outputs = 8,
        model_dim = 16,
        n_blocks = 6,
        n_heads = 8,
        max_length = 2000, #maximum number of loci in a sample
        activation_function = "ReLU")



#Encoder
model = Sequential()
model.add(multihead_layer_test)
model.add(Bidirectional(LSTM(32, return_sequences=True)))
model.add(Conv1D(32, 3, activation="relu", padding="same"))
model.add(Conv1D(32, 3, activation="relu", padding="same"))
model.add(Conv1D(32, 3, activation="relu", padding="same"))
model.add(Conv1D(64, 3, activation="relu", padding="same"))
model.add(Conv1D(128, 3, activation="relu", padding="same"))

#Decoder
model.add(Conv1DTranspose(128, 3, activation="relu", padding="same"))
model.add(Conv1DTranspose(64, 3, activation="relu", padding="same"))
model.add(Conv1DTranspose(32, 3, activation="relu", padding="same"))
model.add(Conv1DTranspose(32, 3, activation="relu", padding="same"))
model.add(Conv1DTranspose(32, 3, activation="relu", padding="same"))
model.add(Conv1D(3, 3, activation="sigmoid", padding="same"))

#Autoencoder
model.compile(optimizer="adam", loss="mse")


#Autoencoder summary
print("Input Shape: ", input_shape)
model.fit(test_input, test_input, epochs=10)

#save model summary
with open('modelsummary.txt', 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))


#show model diagram
#plot_model(model, to_file='demo.png', show_shapes=True)









#RESIDUAL_CONVOLUTION_1D
from ipynb.fs.full.residual_convolution_1D import RESIDUAL_CONVOLUTION_1D

#pre-specify residual_convolution layer parameters
residual_convolution_1D_test = RESIDUAL_CONVOLUTION_1D(
        filters = 64,
        kernel_size = 3,
        strides = 1,
        padding = "same",
        activation = 'relu')
    
#Encoder
model = Sequential()
model.add(multihead_layer_test)
model.add(Bidirectional(LSTM(32, return_sequences=True)))
model.add(residual_convolution_1D_test)
model.add(Conv1D(32, 3, activation="relu", padding="same"))
model.add(Conv1D(32, 3, activation="relu", padding="same"))
model.add(Conv1D(64, 3, activation="relu", padding="same"))
model.add(Conv1D(128, 3, activation="relu", padding="same"))

#Decoder
model.add(Conv1DTranspose(128, 3, activation="relu", padding="same"))
model.add(Conv1DTranspose(64, 3, activation="relu", padding="same"))
model.add(Conv1DTranspose(32, 3, activation="relu", padding="same"))
model.add(Conv1DTranspose(32, 3, activation="relu", padding="same"))
model.add(Conv1DTranspose(32, 3, activation="relu", padding="same"))
model.add(Conv1D(3, 3, activation="sigmoid", padding="same"))

#Autoencoder
model.compile(optimizer="adam", loss="mse")


#Autoencoder summary
print("Input Shape: ", input_shape)
model.fit(test_input, test_input, epochs=10)
#save model summary
with open('modelsummary.txt', 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))


#show model diagram
#plot_model(model, to_file='demo.png', show_shapes=True)

