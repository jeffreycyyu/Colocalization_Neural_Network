#DEPENDENCIES
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
from tensorflow_probability import distributions as tfd
from tensorflow.keras.layers import Input, Embedding, Activation, Add, Conv1D, Conv1DTranspose, LSTM, Layer, LayerNormalization, ReLU, Embedding, Bidirectional, ZeroPadding3D
from keras.preprocessing.sequence import pad_sequences
from keras.backend import temporal_padding
import tensorflow_addons as tfa
from tensorflow.keras.models import Model, Sequential
from keras.utils.vis_utils import plot_model
import edward2 as ed
import warnings


#MULTI-HEAD ATTENTION

class MULTI_HEAD_ATTENTION(tf.keras.layers.Layer):
    """
    Multi-head attention network with positional encoding.
    """
    def __init__(
        self,
        n_traits: int,
        n_outputs: int = 128,
        model_dim: int = 64,
        n_blocks: int = 6,
        n_heads: int = 32,
        activation_function: str = 'ReLU',
        name: str = 'multi_head_attention'
        ):
        """
        Argument(s):
            n_traits: number of traits to be annalyzed (number of channels)
            n_outputs: number of outputs (final layer Conv1D's filter size)
            model_dim: model size (used in calculation of multi-head head_size)
            n_blocks: number of blocks (number of times MULTI_HEAD_ATTENTION_ENCODER.block is called)
            n_heads: number of attention heads (num_heads for tfa.layers.MultiHeadAttention layer)
            activation_function: output activation function (choose from: 'Sigmoid', 'ReLU', 'Linear')
            name: name
        """
        print('MULTI_HEAD_ATTENTION.__init__')
        super().__init__(name = name)
        super(MULTI_HEAD_ATTENTION_ENCODER, self).__init__()
        print('super(MULTI_HEAD_ATTENTION_ENCODER).__init__()')
        
        self._name = name
        
        self.n_traits = n_traits
        self.n_outputs = n_outputs
        self.model_dim = model_dim
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.n_filters = model_dim*4
        self.activation_function = activation_function
        self.head_dim = self.model_dim // self.n_heads

    def block(self, x):
        """
        Multi-head attention block; transformer encoder block.
        Argument(s):
            x: input
        Returns:
            norm_2: multi-head attention block output
        """
        #"Multi-Head Attention" layer
        multihead = tfa.layers.MultiHeadAttention(
            head_size=self.head_dim,
            num_heads=self.n_heads,
            dropout=0.0,
            use_projection_bias=False,
            )([x, x, x]) #use x for key, query, and value to have self-attention

        #first "Add and Norm" layer
        add_1 = Add()([x, multihead])
        norm_1 = LayerNormalization(
            axis=2,
            epsilon=1e-6,
            center=True,
            scale=True
            )(add_1)

        #"Feed Forward" layer
        feed_forward = self.feedforward(norm_1)

        #second "Add and Norm" layer
        add_2 = Add()([norm_1, feed_forward])
        norm_2 = LayerNormalization(
            axis=2,
            epsilon=1e-6,
            center=True,
            scale=True
            )(add_2)
        
        #output: input after transformer block
        return norm_2

    def feedforward(self, x):
        """
        Feedforward network.
        Argument(s):
            x: input
        Returns:
            x: output of the feedforward block
        """
        #feedforward neural network for 1D inputs and ReLU intermediate activation function
        x = Conv1D(self.n_filters, 1, use_bias=True)(x)
        x = ReLU()(x)
        x = Conv1D(self.n_traits, 1, use_bias=True)(x)
        return x
    
    def __call__(self, input_array):
        """
        Argument(s):
            input_array: input
        Returns:
            output_array: output of the multihead attention network
        """
        print('MULTI_HEAD_ATTENTION.__call__')
        
        #zero pad input list of samples
        padded_test_input = pad_sequences(input_array, padding='post', dtype='float32')
        
        #initialize list to store zipped data into
        zipped_padded_test_input = []
        #zip the data to convert 'list of channels which are lists of sequences' into 'lists of sequences which are lists of channels'
        for i in range(int(len(padded_test_input)/self.n_traits)):
            #zip every set of n_traits together (e.g., if 3 traits/channels, then we want the first 3 sequences representing the three channels for the first gene segment to be zipped, then next 3, ...)
            sub_zipped_padded_test_input = [list(l) for l in zip(padded_test_input[0+i*3], padded_test_input[1+i*3], padded_test_input[2+i*3])]
            #append to master list
            zipped_padded_test_input.append(sub_zipped_padded_test_input)
        
        #add postitional encoding
        position_embedding_layer = Embedding(len(zipped_padded_test_input[0]), self.n_traits, mask_zero=True)
        position_indices = tf.range(len(zipped_padded_test_input[0]))
        positional_encoding = position_embedding_layer(position_indices)
        x = zipped_padded_test_input + positional_encoding
        
        #repeat multi-head attention block multiple times
        for _ in range(self.n_blocks): x = self.block(x)
            
        #final convolution
        output_array = Conv1D(self.n_traits, 1, use_bias=True)(x)
            
        #choose final activation function for output
        if self.activation_function == "Sigmoid": output_array = Activation('sigmoid')(output_array)
        elif self.activation_function == "ReLU": output_array = ReLU()(output_array)
        elif self.activation_function == "Linear": output_array = output_array
        else: raise ValueError("Invalid activation_function")
        
        #output: transformer encoder output
        return output_array



#MULTI-HEAD ATTENTION IMPORTANCE WEIGHED VARIATIONAL AUTOENCODER
class MULTI_HEAD_ATTENTION_IMPORTANCE_WEIGHED_VARIATIONAL_AUTOENCODER(tf.keras.Model):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        n_traits: int,
        mha_n_outputs: int = 128,
        mha_model_dim: int = 64,
        mha_n_blocks: int = 6,
        mha_n_heads: int = 32,
        mha_activation_function: str = 'ReLU',
        n_monte_carlo: int = 1,
        n_importance: int = 1,
        name: str = 'multi_head_attention_importance_weighed_variational_autoencoder'
        ):
        """
        Argument(s):
            latent_dim: dimensions of latent space representation
            hidden_dim: dimension of hidden nodes
            
            mha_n_traits: the multi-head attention block's number of traits to be annalyzed (number of channels)
            mha_n_outputs: the multi-head attention block's number of outputs (final layer Conv1D's filter size)
            mha_model_dim: the multi-head attention block's model size (used in calculation of multi-head head_size)
            mha_n_blocks: the multi-head attention block's number of blocks (number of times MULTI_HEAD_ATTENTION_ENCODER.block is called)
            mha_n_heads: the multi-head attention block's number of attention heads (num_heads for tfa.layers.MultiHeadAttention layer)
            mha_activation_function: the multi-head attention block's output activation function (choose from: 'Sigmoid', 'ReLU', 'Linear')
            
            n_monte_carlo: number of Monte Carlo samples for evidence-lower-bound (ELBO) estimation; default is 1 since 1 element is sufficient as long as Monte-Carlo resampling happens at each optimization iteration
            n_importance: number of importance weights
            
            name: name
        """
        print('MULTI_HEAD_ATTENTION_IMPORTANCE_WEIGHED_VARIATIONAL_AUTOENCODER.__init__')
        super().__init__(name = name)
        super(MULTI_HEAD_ATTENTION_IMPORTANCE_WEIGHED_VARIATIONAL_AUTOENCODER, self).__init__()
        print('super(MULTI_HEAD_ATTENTION_IMPORTANCE_WEIGHED_VARIATIONAL_AUTOENCODER).__init__()')
        
        self._name = name
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_traits = n_traits

        self.mha_n_traits = n_traits
        self.mha_n_outputs = mha_n_outputs
        self.mha_model_dim = mha_model_dim
        self.mha_n_blocks = mha_n_blocks
        self.mha_n_heads = mha_n_heads
        self.mha_n_filters = mha_model_dim*4
        self.mha_activation_function = mha_activation_function
        self.mha_head_dim = self.mha_model_dim // self.mha_n_heads
        
        self.n_monte_carlo = n_monte_carlo
        self.n_importance = n_importance
        
    def encoder(self, x):
        
        #build multi-head attention layer model
        multi_head_attention_encoder = MULTI_HEAD_ATTENTION_ENCODER(
            n_traits = self.mha_n_traits,
            n_outputs = self.mha_n_outputs,
            model_dim = self.mha_model_dim,
            n_blocks = self.mha_n_blocks,
            n_heads = self.mha_n_heads,
            activation_function = self.mha_activation_function)
        
        #execute multi-head attention model
        output_mha = multi_head_attention_encoder(x)
        
        x = tf.keras.layers.Dense(self.hidden_dim, activation=tf.nn.relu, dtype=tf.float32)(output_mha)
        x = tf.keras.layers.Dense(self.hidden_dim, activation=tf.nn.relu, dtype=tf.float32)(x)
        mapped = tf.keras.layers.Dense(2*int(self.latent_dim), dtype=tf.float32)(x)
        
        #output: multi-variate normal distributed latent representation
        return tfd.MultivariateNormalDiag(
            loc=mapped[..., :int(self.latent_dim)],
            scale_diag=tf.nn.softplus(mapped[..., int(self.latent_dim):]
            ))

    
    def decoder(self, x): #CHANGE_ME make like a transformers decoder layer
        
        #build model
        layers = tf.keras.Sequential([
            tf.keras.layers.Dense(4*self.latent_dim, activation=tf.nn.relu, dtype=tf.float32), #CHANGE_ME self.hidden_dim to 4*self.latent_dim
            tf.keras.layers.Dense(4*self.latent_dim, activation=tf.nn.relu, dtype=tf.float32), #CHANGE_ME self.hidden_dim to 4*self.latent_dim
            tf.keras.layers.Dense(self.n_traits, dtype=tf.float32) #CHANGE_ME self.hidden_dim to 4*self.latent_dim
            ])(x)
        #output: Bernoulli distributed output
        return tfd.Bernoulli(logits=layers)
    
    
    def assign_prior(self):
        #set prior to be a multivariate normal distribution
        self.prior = tfd.MultivariateNormalDiag(loc=tf.zeros(self.latent_dim, dtype=tf.float32),
                                                scale_diag=tf.ones(self.latent_dim, dtype=tf.float32))
        #output: prior
        return self.prior
    
    
    def compute_loss(self, x):
        # #check if input is in '[batch size, number of loci per segment, channels]' format
        assert len(x.shape) == 3
        #convert to tensor
        x = tf.identity(x)
        #create a tensor by repeating input 'x' M*K-times horizontally; default is no tiling
        x = tf.tile(x, [self.n_monte_carlo * self.n_importance, 1, 1])

        #assign prior distribution
        prior = self.assign_prior()
        #encode input
        encoded = self.encode(x)
        #store latent representation
        latent_representation = encoded.sample()
        #decode latent representation
        decoded = self.decode(latent_representation)

        #compute negative log likelihood
        negative_log_likelihood = -decoded.log_prob(x)
        #if negative loglikelihood is finite, report it, otherwise report all zeros of the same shape
        negative_log_likelihood = tf.where(tf.math.is_finite(negative_log_likelihood), negative_log_likelihood, tf.zeros_like(negative_log_likelihood))
        #reduce along the second and third dimensions
        negative_log_likelihood = tf.reduce_sum(negative_log_likelihood, [1, 2])
        
        #if number of importance weights is greater than 1
        if self.n_importance > 1:
            #compute Kulback-Leiber divergence accounting for number of Monte Carlo samples
            kullback_leibler = encoded.log_prob(latent_representation) - prior.log_prob(latent_representation)
            kullback_leibler = tf.where(tf.is_finite(kullback_leibler), kullback_leibler, tf.zeros_like(kullback_leibler))
            kullback_leibler = tf.reduce_sum(kullback_leibler, 1)
            weights = -negative_log_likelihood - kullback_leibler
            weights = tf.reshape(weights, [self.n_monte_carlo, self.n_importance, -1])
            elbo = tf.log(tf.reduce_mean(tf.exp(weights - tf.reduce_max(weights, axis=1, keepdims=True)), axis=1, keepdims=True) + 1e-5) + tf.reduce_max(weights, axis=1, keepdims=True)
            elbo = tf.reduce_mean(elbo)
        #if number of importance weights is equal to 1
        elif self.n_importance == 1:
            #compute Kulback-Leiber divergence analytically
            kullback_leibler = tfd.kl_divergence(encoded, prior)
            kullback_leibler = tf.where(tf.math.is_finite(kullback_leibler), kullback_leibler, tf.zeros_like(kullback_leibler))
            kullback_leibler = tf.reduce_sum(kullback_leibler, 1)
        else: raise ValueError('n_importance should be a positive integer')

        #compute evidence lower bound as negative log likelihood multiplied by Kulback-Leiber divergence
        elbo = negative_log_likelihood * kullback_leibler
        #compute evidence lower bound mean across all tensor dimensions
        elbo = tf.reduce_mean(elbo)
        
        #output: evidence lower bound loss
        return elbo
    
    
    def __call__(self, inputs, training=False):
        #first encode, then sample from encoded latent representation distribution, then decode, finally sample from decoded reconstruction distribution
        return self.decoder(self.encoder(inputs).sample()).sample()   
    

#TEST INPUT

#set random seed
random.seed(25252)

#data file pathname
pathname = '/Users/jeffreyyu/Documents/Sladek/colocalization_neural_network/data/simulated_gwas.txt'

n_samples = 40
#import datafram from txt file
test_dataframe = pd.read_csv(pathname, sep='\t')
#remove heading name
test_dataframe.columns.name = None

#convert to float 32 numpy array
data_frame_array = df.to_numpy(test_dataframe, dtype = 'float32')  

#delete position column
no_positions_array = np.delete(data_frame_array, 0, 1)

no_positions_array = no_positions_array.T

#cuts positions to split the dataset
#split_sample_input represents a list of variable length simulated GWAS p-values with 100 samples, 3 traits for each loci (variable number of loci per sample)
cuts = np.sort(random.sample(range(0, data_frame_array.shape[0]-1), n_samples-1))

test_input = np.split(no_positions_array , cuts, axis = 1)


print(test_input[0].shape)
print(test_input[1].shape)
print(len(test_input))


#turn from 40 samples to 120 samples (disjoin all 3 traits)
new_list = [e for sl in test_input for e in sl]
#convert from list of 1d arrays to list of lists
new_list = [l.tolist() for l in new_list]
print(len(new_list))
#these 3 from the same sample so 3 traits shoudl have same number of loci
print(len(new_list[0]))
print(len(new_list[1]))
print(len(new_list[2]))
#these are from sample #2 so should have different number of loci than before
print(len(new_list[3]))
print(len(new_list[4]))
print(len(new_list[5]))


#RAW TEST

def testing(x):
    
    #ENCODER-ENCODER-ENCODER-ENCODER-ENCODER-ENCODER-ENCODER-ENCODER-ENCODER-ENCODER-ENCODER-ENCODER-ENCODER-ENCODER-ENCODER-ENCODER-ENCODER-ENCODER-ENCODER
    hidden_dim = 64
    latent_dim = 32
    n_outputs = 3
    
    
    multi_head_attention_encoder = MULTI_HEAD_ATTENTION_ENCODER(
        n_traits = 3,
        n_outputs = 64,
        model_dim = 16,
        n_blocks = 6,
        n_heads = 8,
        activation_function = 'ReLU')
    print('master mha1') #DELETE_ME
    output_mha = multi_head_attention_encoder(x)
    print('master mha2') #DELETE_ME


    x = tf.keras.layers.Dense(hidden_dim, activation=tf.nn.relu, dtype=tf.float32)(output_mha)
    print('master dense1') #DELETE_ME
    x = tf.keras.layers.Dense(hidden_dim, activation=tf.nn.relu, dtype=tf.float32)(x)
    print('master dense2') #DELETE_ME
    mapped = tf.keras.layers.Dense(2*int(latent_dim), dtype=tf.float32)(x)
    print('master dense3') #DELETE_ME
    

    #output: multi-variate normal distributed latent representation
    final_encoder = tfd.MultivariateNormalDiag(
        loc=mapped[..., :int(latent_dim)],
        scale_diag=tf.nn.softplus(mapped[..., int(latent_dim):]
        ))
    print('master MultivariateNormalDiag') #DELETE_ME
    
    sample_final_encoder = final_encoder.sample()
    print('master sample_final_encoder')
    
    #DECODER-DECODER-DECODER-DECODER-DECODER-DECODER-DECODER-DECODER-DECODER-DECODER-DECODER-DECODER-DECODER-DECODER-DECODER-DECODER-DECODER-DECODER-DECODER
    
    x = sample_final_encoder
    
    tf.keras.layers.Dense(4*latent_dim, activation=tf.nn.relu, dtype=tf.float32)(x)
    print('master de-dense1') #DELETE_ME
    tf.keras.layers.Dense(4*latent_dim, activation=tf.nn.relu, dtype=tf.float32)(x)
    print('master de-dense2') #DELETE_ME
    layers = tf.keras.layers.Dense(n_traits, dtype=tf.float32)(x)
    print('master de-dense3') #DELETE_ME
    
    final_decoder = tfd.Bernoulli(logits=layers)
    print('master Bernoulli') #DELETE_ME
    
    sample_final_decoder = final_decoder.sample()
    print('master sample_final_decoder')
    
    return sample_final_decoder

output = testing(test_input)
print(output)


#FUNCTIONAL TEST

test_full_function = MULTI_HEAD_ATTENTION_IMPORTANCE_WEIGHED_VARIATIONAL_AUTOENCODER(32, 18, 3)

test_full_function_out = test_full_function(test_input)

print(test_full_function_out)

