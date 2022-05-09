class MULTI_HEAD_ATTENTION(tf.keras.layers.Layer):
    """
    Multi-head attention network with positional encoding (BERT).
    """
    def __init__(
        self,
        n_outputs: int,
        model_dim: int,
        n_blocks: int,
        n_heads: int,
        max_length: int,
        activation_function: str,
        name: str = 'multi_head_attention'
        ):
        """
        Argument(s):
            n_outputs: number of outputs
            model_dim: model size
            n_blocks: number of blocks
            n_heads: number of attention heads
            max_length: maximum length for positional encoding
            activation_function: output activation function
            name: name
        """
        print('MULTI_HEAD_ATTENTION.__init__')
        super().__init__(name = name)
        super(MULTI_HEAD_ATTENTION, self).__init__()

        self.n_outputs = n_outputs
        self.model_dim = model_dim
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.n_filters = model_dim*4
        self.max_length = max_length
        self.activation_function = activation_function
        self.head_dim = self.model_dim // self.n_heads

    def block(self, x):
        """
        Multi-head attention block.
        Argument(s):
            x: input
        Returns:
            layer_normalized: multi-head attention block output
        """
        layer = tfa.layers.MultiHeadAttention(
            head_size=self.head_dim,
            num_heads=self.n_heads,
            output_size=self.model_dim,
            dropout=0.0,
            use_projection_bias=False,
            )([x, x, x]) #use x for key, query, and value to have self-attention
        layer = Add()([x, layer])
        layer = LayerNormalization(
            axis=2,
            epsilon=1e-6,
            center=True,
            scale=True
            )(layer)

        layer_forward = self.feedforward(layer)
        layer_summed = Add()([layer, layer_forward])
        layer_normalized = LayerNormalization(
            axis=2,
            epsilon=1e-6,
            center=True,
            scale=True
            )(layer_summed)
        return layer_normalized

    def feedforward(self, x):
        """
        Feedforward network.
        Argument(s):
            x: input
        Returns:
            x: output of the feedforward block
        """
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

        ## Add postitional encoding.
        position_idx = tf.tile([tf.range(tf.shape(x)[1])], [tf.shape(x)[0], 1])
        positional_encoding = Embedding(self.max_length, self.model_dim)(position_idx)
        x = Add()([x, positional_encoding])

        for _ in range(self.n_blocks): x = self.block(x)

        self.output_array = Conv1D(self.n_outputs, 1, use_bias=True)(x)

        if self.activation_function == "Sigmoid": self.output_array = Activation('sigmoid')(self.output_array)
        elif self.activation_function == "ReLU": self.output_array = ReLU()(self.output_array)
        elif self.activation_function == "Linear": self.output_array = self.output_array
        else: raise ValueError("Invalid activation_function")
        
        return self.output_array
    
