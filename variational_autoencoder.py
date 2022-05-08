class VARIATIONAL_AUTOENCODER(tf.keras.Model):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        n_monte_carlo: int = 1,
        n_importance: int = 1,
        name: str = 'variational_autoencoder'
        ):
        """
        Argument(s):
            input_dim: dimensions of input data
            hidden_dim: dimensions of hidden layer filters (for all hiden layers)
            latent_dim: dimensions of latent space representation
            n_monte_carlo: number of Monte Carlo samples for evidence-lower-bound (ELBO) estimation
            n_importance: number of importance weights
            name: name
        """
        print('AUGMENT_DATA.__init__')
        super().__init__(name = name)
        super(VARIATIONAL_AUTOENCODER, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.n_monte_carlo = n_monte_carlo
        self.n_importance = n_importance
        self._name = name
        
        
    def encoder(self, x):
        #convert to tensor
        x = tf.identity(x)
        #build model
        mapped = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_dim, activation=tf.nn.relu, dtype=tf.float32),
            tf.keras.layers.Dense(self.hidden_dim, activation=tf.nn.relu, dtype=tf.float32),
            tf.keras.layers.Dense(2*int(self.latent_dim), dtype=tf.float32)
            ])(x)
        #output: multi-variate normal distributed latent representation
        return tfd.MultivariateNormalDiag(
            loc=mapped[..., :int(self.latent_dim)],
            scale_diag=tf.nn.softplus(mapped[..., int(self.latent_dim):]
            ))

    
    def decoder(self, x):
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
