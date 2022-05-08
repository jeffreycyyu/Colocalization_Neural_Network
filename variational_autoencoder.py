class VARIATIONAL_AUTOENCODER(tf.keras.Model):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        name: str = 'variational_autoencoder'
        ):
        """
        Argument(s):
            input_dim: dimensions of input data
            hidden_dim: dimensions of hidden layer filters (for all hiden layers)
            latent_dim: dimensions of latent space representation
            name: name
        """
        print('AUGMENT_DATA.__init__')
        super().__init__(name = name)
        super(VAE, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self._name = name
        
    def encoder(self, x):
        x = tf.identity(x)
        mapped = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_dim, activation=tf.nn.relu, dtype=tf.float32),
            tf.keras.layers.Dense(self.hidden_dim, activation=tf.nn.relu, dtype=tf.float32),
            tf.keras.layers.Dense(2*int(self.latent_dim), dtype=tf.float32)
            ])(x)
        return tfd.MultivariateNormalDiag(
            loc=mapped[..., :int(self.latent_dim)],
            scale_diag=tf.nn.softplus(mapped[..., int(self.latent_dim):]
            ))

    def decoder(self, x):
        x = tf.identity(x)
        layers = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_dim, activation=tf.nn.relu, dtype=tf.float32),
            tf.keras.layers.Dense(self.hidden_dim, activation=tf.nn.relu, dtype=tf.float32),
            tf.keras.layers.Dense(self.input_dim, dtype=tf.float32)
            ])(x)                       
        return tfd.Bernoulli(logits=layers)
    
    def __call__(self, inputs):
        return self.decoder(self.encoder(inputs).sample()).sample()
    
    
    

print(test_input)
test_VAE = VARIATIONAL_AUTOENCODER(60, 919)
print(test_VAE(test_input))

    
