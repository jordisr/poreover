import tensorflow as tf
import numpy as np

# Positional encoding code from https://www.tensorflow.org/tutorials/text/transformer
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

class transformer(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff, num_heads, key_dim, **kwargs):
        super(transformer, self).__init__(**kwargs)
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.key_dim = key_dim

    def build(self, input_shape):
        self.attention_layer = tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dim)
        self.ff_inner_layer = tf.keras.layers.Dense(self.d_ff, activation='relu')
        self.ff_outer_layer = tf.keras.layers.Dense(self.d_model, activation=None)
        self.layer_norm_attention = tf.keras.layers.LayerNormalization()
        self.layer_norm_ff = tf.keras.layers.LayerNormalization()

    def call(self, inputs):
        # attention block with residual connection
        x = inputs + self.attention_layer(inputs, inputs)
        x = self.layer_norm_attention(x)

        # ff block with residual connection
        x = x + self.ff_outer_layer(self.ff_inner_layer(inputs))
        x = self.layer_norm_ff(x)

        return x

    # allow for serialization
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'd_model': self.d_model,
            'd_ff': self.d_ff,
            'num_heads': self.num_heads,
            'key_dim': self.key_dim
        })
        return config
