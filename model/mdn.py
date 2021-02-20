import tensorflow as tf


class MDN(tf.keras.layers.Layer):
    mdn_w: tf.Variable
    mdn_b: tf.Variable

    def __init__(self, rnn_size, n_mixtures, initializer, **kwargs):
        super(MDN, self).__init__(**kwargs)
        self.rnn_size = rnn_size
        self.n_mixtures = n_mixtures
        self.initializer = initializer

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'rnn_size': self.rnn_size,
            'n_mixtures': self.n_mixtures,
            'initializer': self.initializer,
        })
        return config

    def build(self, input_shape):
        n_out = 1 + self.n_mixtures * 6  # params = end_of_stroke + 6 parameters per Gaussian
        self.mdn_w = self.add_weight("output_w", shape=[self.rnn_size, n_out], initializer=self.initializer)
        self.mdn_b = self.add_weight("output_b", shape=[n_out], initializer=self.initializer)

    def call(self, input0, **kwargs):
        print(input0)
        print(self.mdn_w)
        print(self.mdn_b)
        flattened = tf.reshape(tf.concat(input0, 0), [-1, self.rnn_size])  # concat outputs for efficiency
        print(flattened)
        dense = tf.add(tf.matmul(flattened, self.mdn_w), self.mdn_b)
        print(dense)
        return tf.stack(dense)