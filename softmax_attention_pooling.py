#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Activation, Add, Conv1D, Conv1DTranspose, LSTM, Layer, LayerNormalization, ReLU, Embedding, Bidirectional
import tensorflow_addons as tfa
from tensorflow.keras.models import Model, Sequential
from keras.utils.vis_utils import plot_model


class SOFTMAX_ATTENTION_POOLING_1D(tf.keras.layers.Layer):
    """
    1-Dimensional softmax pooling operation with optional weights.
    """

    def __init__(
        self,
        pool_size: int,
        w_init_scale: float,
        name: str = 'softmax_attention_pooling_1D'
    ):
        """
        Argument(s):
            pool_size: pooling size
            per_channel: If True, the logits/softmax weights will be computed for
                each channel separately. If False, same weights will be used across all
                channels.
            w_init_scale: When 0.0 is equivalent to avg pooling, and when
                ~2.0 and `per_channel=False` it's equivalent to max pooling.
            name: name
        """
        super().__init__(name = name)
        self._pool_size = pool_size
        self._per_channel = per_channel
        self._w_init_scale = w_init_scale
        self._logit_linear = None

    def _initialize(self, n_features):
         """
        Argument(s):
            n_features: number of feature channels
        """
        self._logit_linear = snt.Linear(
                output_size=n_features if self._per_channel else 1,
                with_bias=False,    # Softmax is agnostic to shifts.
                w_init=snt.initializers.Identity(self._w_init_scale))

    def __call__(self, inputs):
        """
        Argument(s):
            inputs: input
        """
        _, length, n_features = inputs.shape
        self._initialize(n_features)
        inputs = tf.reshape(
                inputs,
                (-1, length // self._pool_size, self._pool_size, n_features))
        return tf.reduce_sum(
                inputs * tf.nn.softmax(self._logit_linear(inputs), axis=-2),
                axis=-2)

