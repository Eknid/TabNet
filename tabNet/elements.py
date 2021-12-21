import tensorflow as tf
import tensorflow_addons as tfa

class GLU(tf.keras.layers.Layer):
    def __init__(self, dim, momentum, initializer=tf.keras.initializers.GlorotNormal(), **kwargs):
        super(GLU, self).__init__(**kwargs)
        self.dim = dim
        self.momentum = momentum
        self.fcn = tf.keras.layers.Dense(2*self.dim, kernel_initializer=initializer)
        self.bn = tf.keras.layers.BatchNormalization(momentum=self.momentum)
    
    def get_config(self):
        config = super(GLU, self).get_config()
        config['dim'] = self.dim
        config['momentum'] = self.momentum
        return config

    def call(self, inp, training=None):
        x = self.fcn(inp)
        x = self.bn(x, training=training)
        return x[..., :self.dim] * tf.nn.sigmoid(x[..., self.dim:])

class FeatureTransformerShared(tf.keras.layers.Layer):
    def __init__(self, dim, len=2, norm=tf.sqrt(0.5), momentum=0.9, **kwargs):
        super(FeatureTransformerShared, self).__init__(**kwargs)
        self.dim = dim
        self.len = len
        self.norm = norm
        self.momentum = momentum
        self.glu = [GLU(dim=self.dim, momentum=self.momentum) for i in range(len)]

    def get_config(self):
        config = super(FeatureTransformerShared, self).get_config()
        config['dim'] = self.dim
        config['len'] = self.len
        config['momentum'] = self.momentum
        return config

    def call(self, inp, training=None):
        x = self.glu[0](inp, training=training)

        for i in range(1, self.len):
            y = self.glu[i](x, training=training)
            x = (x + y) * self.norm
        return x

class FeatureTransformerStep(tf.keras.layers.Layer):
    def __init__(self, dim, len=2, norm=tf.sqrt(0.5), momentum=0.9, **kwargs):
        super(FeatureTransformerStep, self).__init__(**kwargs)
        self.dim = dim
        self.len = len
        self.norm = norm
        self.momentum = momentum
        self.glu = [GLU(dim=self.dim, momentum=self.momentum) for i in range(len)]

    def get_config(self):
        config = super(FeatureTransformerStep, self).get_config()
        config['dim'] = self.dim
        config['len'] = self.len
        config['momentum'] = self.momentum
        return config

    def call(self, inp, training=None):
        x = self.glu[0](inp, training=training)
        y = (x + inp) * self.norm

        for i in range(1, self.len):
            x = self.glu[i](y, training=training)
            y = (x + y) * self.norm
        return y

class AttentiveTransformer(tf.keras.layers.Layer):
    def __init__(self, dim, momentum=0.9, initializer=tf.keras.initializers.GlorotNormal(), **kwargs):
        super(AttentiveTransformer, self).__init__(**kwargs)
        self.dim = dim
        self.momentum = momentum
        self.fcn = tf.keras.layers.Dense(dim, use_bias=False, kernel_initializer=initializer)
        self.bn = tf.keras.layers.BatchNormalization(momentum=self.momentum)


    def get_config(self):
        config = super(AttentiveTransformer, self).get_config()
        config['dim'] = self.dim
        config['momentum'] = self.momentum
        return config

    def call(self, inp, priors, training=None):
        x = self.fcn(inp)
        x = self.bn(x, training=training)
        mask = tfa.activations.sparsemax(priors * x)
        return mask