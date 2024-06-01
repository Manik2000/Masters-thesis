import tensorflow as tf
from keras.activations import softmax, tanh
from keras.layers import Dense, Layer


class Attention(Layer):

    def __init__(self, units, name="attention", **kwargs):
        super(Attention, self).__init__(name=name, **kwargs)
        self.W = Dense(units, use_bias=True)
        self.V = Dense(1)

    def call(self, inputs):
        # Compute attention scores
        score = tanh(self.W(inputs))
        attention_weights = softmax(self.V(score), axis=1)

        # Apply attention weights to input
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector

    def build(self, input_shape):
        self.W.build(input_shape)
        self.V.build(input_shape)
        self.built = True
