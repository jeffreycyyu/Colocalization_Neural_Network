#MULTI-HEAD ATTENTION IMPORTANCE WEIGHED VARIATIONAL AUTOENCODER
class MULTI_HEAD_ATTENTION_ENCODER(tf.keras.layers.Layer):
    """
    Multi-head attention network with positional encoding (BERT).
    """
    def __init__(
        self,
        max_length: int,
        n_outputs: int = 128,
        model_dim: int = 64,
        n_blocks: int = 6,
        n_heads: int = 32,
        activation_function: str = 'ReLU',
        name: str = 'multi_head_attention'
        ):
        """
        Argument(s):
            max_length: maximum length for positional encoding (maximum number of loci across all samples)
            n_outputs: number of outputs (final layer Conv1D's filter size)
            model_dim: model size for multihead attention output (output_size for tfa.layers.MultiHeadAttention layer)
            n_blocks: number of blocks (number of times MULTI_HEAD_ATTENTION_ENCODER.block is called)
            n_heads: number of attention heads (num_heads for tfa.layers.MultiHeadAttention layer)
            activation_function: output activation function (choose from: 'Sigmoid', 'ReLU', 'Linear')
            name: name
        """
        print('MULTI_HEAD_ATTENTION.__init__')
        super().__init__(name = name)
        super(MULTI_HEAD_ATTENTION, self).__init__()
        
        self._name = name
        
        self.max_length = max_length
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
            layer_normalized: multi-head attention block output
        """
        #"Multi-Head Attention" layer
        multihead = tfa.layers.MultiHeadAttention(
            head_size=self.head_dim,
            num_heads=self.n_heads,
            output_size=self.model_dim,
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
        x = Conv1D(self.model_dim, 1, use_bias=True)(x)
        return x
    
    def __call__(self, input_array):
        """
        Argument(s):
            input_array: input
        Returns:
            self.output_array: output of the multihead attention network
        """
        print('MULTI_HEAD_ATTENTION.__call__')
        
        x = Conv1D(self.model_dim, 1, use_bias=False)(input_array)
        x = LayerNormalization(axis=2, epsilon=1e-6, center=True, scale=True)(x)
        x = ReLU()(x)
        
        #pad input list of samples 
        padded_test_input = pad_sequences(x, padding='post', dtype='float32')
        
        #add postitional encoding
        position_idx = tf.tile([tf.range(tf.shape(x)[1])], [tf.shape(x)[0], 1])
        positional_encoding = Embedding(self.max_length, self.model_dim)(position_idx)
        x = Add()([x, positional_encoding])

        #repeat multi-head attention block multiple times
        for _ in range(self.n_blocks): x = self.block(x)
        
        #final convolution
        output_array = Conv1D(self.n_outputs, 1, use_bias=True)(x)
        
        #choose final activation function for output
        if self.activation_function == "Sigmoid": output_array = Activation('sigmoid')(output_array)
        elif self.activation_function == "ReLU": output_array = ReLU()(output_array)
        elif self.activation_function == "Linear": output_array = output_array
        else: raise ValueError("Invalid activation_function")
        
        #output: transformer encoder output
        return output_array
    
    
    
class MULTI_HEAD_ATTENTION_IMPORTANCE_WEIGHED_VARIATIONAL_AUTOENCODER(tf.keras.Model):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        mha_max_length: int,
        mha_n_outputs: int = 128,
        mha_model_dim: int = 64,
        mha_n_blocks: int = 6,
        mha_n_heads: int = 8,
        mha_activation_function: str = 'ReLU',
        n_monte_carlo: int = 1,
        n_importance: int = 1,
        name: str = 'multi_head_attention_importance_weighed_variational_autoencoder'
        ):
        """
        Argument(s):
            input_dim: dimensions of input data
            latent_dim: dimensions of latent space representation
            
            mha_max_length: maximum length for positional encoding (maximum number of loci across all samples)
            mha_n_outputs: number of outputs (final layer Conv1D's filter size)
            mha_model_dim: model size for multihead attention output (output_size for tfa.layers.MultiHeadAttention layer)
            mha_n_blocks: number of blocks (number of times MULTI_HEAD_ATTENTION_ENCODER.block is called)
            mha_n_heads: number of attention heads (num_heads for tfa.layers.MultiHeadAttention layer)
            mha_activation_function: output activation function (choose from: 'Sigmoid', 'ReLU', 'Linear')
            
            n_monte_carlo: number of Monte Carlo samples for evidence-lower-bound (ELBO) estimation
            n_importance: number of importance weights
            
            name: name
        """
        print('AUGMENT_DATA.__init__')
        super().__init__(name = name)
        super(MULTI_HEAD_ATTENTION_IMPORTANCE_WEIGHED_VARIATIONAL_AUTOENCODER, self).__init__()

        self._name = name
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.mha_max_length = mha_max_length
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
        #convert to tensor
        x = tf.identity(x)
        
        #build model
        multi_head_attention_encoder = MULTI_HEAD_ATTENTION_ENCODER(
            max_length = self.mha_max_length,
            n_outputs = self.mha_n_outputs,
            model_dim = self.mha_model_dim,
            n_blocks = self.mha_n_blocks,
            n_heads = self.mha_n_heads,
            activation_function = self.mha_activation_function)
        mapped = multi_head_attention_encoder(x)
        
        #output: multi-variate normal distributed latent representation
        return tfd.MultivariateNormalDiag(
            loc=mapped[..., :int(self.latent_dim)],
            scale_diag=tf.nn.softplus(mapped[..., int(self.latent_dim):]
            ))

    
    def decoder(self, x): #CHANGE_ME make like a transformers decoder layer
        #convert to tensor
        x = tf.identity(x)
        #build model
        layers = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_dim, activation=tf.nn.relu, dtype=tf.float32),
            tf.keras.layers.Dense(self.hidden_dim, activation=tf.nn.relu, dtype=tf.float32),
            tf.keras.layers.Dense(self.input_dim, dtype=tf.float32)
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
        #check if input is in '[batch size, number of loci per segment, channels]' format
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
    
    
    
    

    
    
    
