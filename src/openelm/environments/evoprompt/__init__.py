# Below are the four seeds provided
# by the original paper EvoPrompting: Language Models for Code-Level Neural Architecture Search
# from https://arxiv.org/pdf/2302.14838.pdf


HAND_DESIGNED = """
class Model(nn.Module):
    features: int = 32
    nlayer: int = 3

    @nn.compact
    def __call__(self, x):
        x = x[..., None]
        x = nn.Conv(features=self.features, kernel_size=(3,))(x)
        x = nn.relu(x)

        x = nn.avg_pool(x, window_shape=(2,), strides=(2,))
        for _ in range(self.nlayer - 1):
            xp = nn.Conv(
            features=self.features,
            kernel_size=(3,),
            )(x)
            xp = nn.relu(xp)
            x = x + xp

    x = nn.avg_pool(x, window_shape=(2,), strides=(2,))
    x = x.reshape((x.shape[0], -1)) # flatten
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)
    x = nn.Dense(features=10)(x)
    return x

"""

FLAX_IMPLEMENTATION_BASELINE = """
class Model(nn.Module):
    features: int = 25

    @nn.compact
    def __call__(self, x):
        x = x[..., None]
        x = nn.Conv(
            features=self.features, kernel_size=(5,), strides=(2,), padding=(1,)
        )(x)
        x = nn.relu(x)
        for _ in range(2):
            x = nn.Conv(
                features=self.features, kernel_size=(3,), strides=(2,), padding=(1,)
                )(x)
                x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=10)(x)
        return x
"""
FLAX_IMPLEMENTATION_OF_GRU = """
class Model(nn.Module):
    \"\"\" A simple GRU model.\"\"\"

    hidden_size: int = 6
    seed: int = 42

    @nn.compact
    def __call__(self, x):
        x = jnp.expand_dims(x, -1)
        rng = jax_random.PRNGKey(self.seed)
        gru = recurrent.GRU(
            hidden_size=self.hidden_size,
            num_layers=1,
            dropout_rate=0.0,
            bidirectional=True,
        )
        lengths = np.full([x.shape[0]], x.shape[1])
        initialized_params = gru.init(rng, x, lengths)
        params = initialized_params[’params’]
        outputs, _ = gru.apply({’params’: params}, x, lengths)
        outputs = outputs.reshape((outputs.shape[0], -1))
        x = nn.Dense(features=10)(outputs)
        return x
"""

FULLY_CONNECTED_BASELINE = """
class Model(nn.Module):
    hidden_size: int = 100

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.hidden_size)(x)
        x = nn.relu(x)
        x = x + nn.relu(nn.Dense(features=self.hidden_size)(x))
        x = nn.Dense(features=10)(x)
        return x

 return Model
"""
