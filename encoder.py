import tensorflow as tf
from tabNet.elements import AttentiveTransformer, FeatureTransformerStep, FeatureTransformerShared

class TabEncoderBlock(tf.keras.layers.Layer):
    def __init__(self, 
                 dim_imp, 
                 dim_d, 
                 dim_a, 
                 shared_dec_steps, 
                 num_dec_steps=2, 
                 momentum=0.9, 
                 split=True, 
                 activation=tf.keras.layers.ReLU(),
                 **kwargs):
        super(TabEncoderBlock, self).__init__(**kwargs)
        self.dim_imp = dim_imp
        self.dim_d = dim_d
        self.dim_a = dim_a
        self.split = split
        self.num_dec_steps = num_dec_steps
        self.momentum = momentum
        self.activation = activation
        self.attn_trans = AttentiveTransformer(self.dim_imp, momentum=self.momentum)
        self.shared_dec_steps = shared_dec_steps
        self.dec_steps = FeatureTransformerStep(self.dim_d + self.dim_a, self.num_dec_steps, momentum=self.momentum)
        

    def get_config(self):
        config = super(TabEncoderBlock, self).get_config()
        config['dim_imp'] = self.dim_imp
        config['dim_d'] = self.dim_d
        config['dim_a'] = self.dim_a
        config['split'] = self.split
        config['momentum'] = self.momentum
        config['num_dec_steps'] = self.num_dec_steps
        return config

    def call(self, inp, a, priors, training=None):
        mask = self.attn_trans(a, priors, training=training)
        y = mask * inp
        y = self.shared_dec_steps(y, training=training)
        y = self.dec_steps(y, training=training)
        if self.split:
            d, a = tf.split(y, num_or_size_splits=[self.dim_d, self.dim_a], axis=-1)
        else:
            d = y
            a = y
        d = self.activation(d)      
        return d, a, mask


class TabEncoder(tf.keras.Model):
    def __init__(self, 
                 dim_imp, 
                 dim_d, 
                 dim_a, 
                 n_steps, 
                 num_shared_dec_steps, 
                 num_dec_steps, 
                 gamma=1.2, 
                 epsilon=1e-5, 
                 sparsity_coef=1e-5, 
                 momentum=0.9, 
                 split=True, 
                 **kwargs):
        super(TabEncoder, self).__init__(**kwargs)
        self.dim_imp = dim_imp
        self.dim_d = dim_d
        self.dim_a = dim_a
        self.n_steps = n_steps
        self.num_shared_dec_steps = num_shared_dec_steps
        self.num_dec_steps = num_dec_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.sparsity_coef = sparsity_coef
        self.momentum = momentum
        self.split = split
        self.bn = tf.keras.layers.BatchNormalization(momentum=self.momentum)
        self.shared_dec_steps = FeatureTransformerShared(self.dim_d + self.dim_a, self.num_shared_dec_steps, momentum=self.momentum)
        self.dec_steps = FeatureTransformerStep(self.dim_d + self.dim_a, self.num_dec_steps, momentum=self.momentum)
        self.tabEncoderBlocks = [TabEncoderBlock(self.dim_imp, self.dim_d, self.dim_a, self.shared_dec_steps, self.num_dec_steps, self.momentum, self.split) for i in range(self.n_steps)]
        

    def get_config(self):
        config = super(TabEncoder, self).get_config()
        config['dim_imp'] = self.dim_imp
        config['dim_d'] = self.dim_d
        config['dim_a'] = self.dim_a
        config['n_steps'] = self.n_steps
        config['num_dec_steps'] = self.num_dec_steps
        config['num_shared_dec_steps'] = self.num_shared_dec_steps
        config['gamma'] = self.gamma
        config['epsilon'] = self.epsilon
        config['sparsity_coef'] = self.sparsity_coef
        config['split'] = self.split
        config['momentum'] = self.momentum
        return config

    def call(self, inp, training=None):
        x = self.bn(inp, training=True)
        y = self.shared_dec_steps(x, training=training)
        y = self.dec_steps(y, training=training)
        if self.split:
            _, a = tf.split(y, num_or_size_splits=[self.dim_d, self.dim_a], axis=-1)
        else:
            a = y

        priors = 1
        output = 0
        importance_numer = 0
        importance_denom = 0
        entropy_loss = 0
        
        for step in range(self.n_steps):
            d, a, mask = self.tabEncoderBlocks[step](x, a, priors, training=training)
            priors = priors * (self.gamma - mask) 
            output = output + d

            eta = tf.reduce_sum(d, axis=-1, keepdims=True)
            scale_agg = eta * mask
            importance_numer = importance_numer + scale_agg
            importance_denom = importance_denom + tf.reduce_sum(eta * tf.cast(tf.math.square(mask), tf.float32), axis=-1, keepdims=True)


            aggregated_mask_loss = tf.reduce_sum(-mask * tf.math.log(mask + self.epsilon), axis=-1)
            entropy_loss += tf.reduce_mean(aggregated_mask_loss) / tf.cast(self.n_steps - 1, tf.float32)
        self.add_loss(self.sparsity_coef * entropy_loss)
             
        return output, importance_numer/importance_denom