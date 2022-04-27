#!/usr/bin/env python
# coding: utf-8

# In[32]:


import os
import math, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Activation, Add, Conv1D, Conv1DTranspose, LSTM, Layer, LayerNormalization, ReLU, Embedding, Bidirectional
import tensorflow_addons as tfa
from tensorflow.keras.models import Model, Sequential
from keras.utils.vis_utils import plot_model

#Model Input
#generate data 400 gene segments each of length 100bp/loci and with 5 traits/cell-types
#test: GWAS, eQTL1, eQTL2, eQTL3, eQTL4 p-values for 100 bp's (100 loci/positions)
#randomly generate 5 vectors which represent the 5 trait's p-values for 100 loci each
specify_input_shape = (400, 100, 5)
test_input = np.random.random(specify_input_shape)


# In[33]:


#MULTI_HEAD_ATTENTION
from ipynb.fs.full.multi_head_attention import MULTI_HEAD_ATTENTION

#pre-specify multi-head attention layer parameters
multihead_layer_test = MULTI_HEAD_ATTENTION(
        n_outputs = 8,
        model_dim = 16,
        n_blocks = 6,
        n_heads = 8,
        max_length = 200,
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
model.add(Conv1D(5, 3, activation="sigmoid", padding="same"))

#Autoencoder
model.compile(optimizer="adam", loss="mse")


#Autoencoder summary
print("Input Shape: ", specify_input_shape)
model.fit(test_input, test_input, epochs=10)
#save model summary
with open('modelsummary.txt', 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))


#show model diagram
#plot_model(model, to_file='demo.png', show_shapes=True)


# In[42]:


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
model.add(Conv1D(5, 3, activation="sigmoid", padding="same"))

#Autoencoder
model.compile(optimizer="adam", loss="mse")


#Autoencoder summary
print("Input Shape: ", specify_input_shape)
model.fit(test_input, test_input, epochs=10)
#save model summary
with open('modelsummary.txt', 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))


#show model diagram
#plot_model(model, to_file='demo.png', show_shapes=True)


# In[ ]:





# In[ ]:




