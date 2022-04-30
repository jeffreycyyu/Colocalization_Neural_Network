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

from ipynb.fs.full.multi_head_attention import MULTI_HEAD_ATTENTION



class RESIDUAL_CONVOLUTION_1D(tf.keras.layers.Layer):
    """
    1-Dimensional residual convolution with activation applied.
    """
    def __init__(
        self,
        filters: int,
        kernel_size: int,
        strides: int,
        padding: str,
        activation: str,
        name: str = 'residual_convolution_1D',
        ):
        print('RESIDUAL_CONVOLUTION_1D.__call__')
        """
        Argument(s):
        filters: dimensionality of the output space
        kernel_size: length of the convolution window,
        strides: stride length of the convolution,
        padding: padding method (valid, same, causal)
        activation: activation function
        name: name
        """
        print('RESIDUAL_CONVOLUTION_1D.__init__')
        super().__init__(name = name)
        super(RESIDUAL_CONVOLUTION_1D, self).__init__()
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        
    def __call__(self, input_array):
        """
        Argument(s):
            input_array: input
        """
        initial_x = input_array
        
        #Convolution
        x = Conv1D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            )(initial_x)
        
        #Add initial input
        Add()([x, initial_x])
        
        #Pass through an activation
        Activation(self.activation)(x)
        
        return x


