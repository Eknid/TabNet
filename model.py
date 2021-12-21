import tensorflow as tf
from tabNet.encoder import TabEncoder

class TabNet(tf.keras.Model):
    def __init__(self, 
                 input_df, 
                 num_features, 
                 cat_features, 
                 dim_d, 
                 dim_a, 
                 num_shared_dec_steps, 
                 num_dec_steps, 
                 n_steps, 
                 n_out, 
                 activation, 
                 gamma=1.2, 
                 epsilon=1e-5, 
                 sparsity_coef=1e-5, 
                 momentum=0.9, 
                 split=True, 
                 string_emb_dim=None, 
                 initializer=tf.keras.initializers.GlorotNormal(),
                 **kwargs):
        super(TabNet, self).__init__(**kwargs) 
        self.dim_d = dim_d
        self.dim_a = dim_a
        self.num_shared_dec_steps = num_shared_dec_steps
        self.num_dec_steps = num_dec_steps
        self.n_steps = n_steps
        self.n_out = n_out
        self.activation = activation
        self.gamma = gamma
        self.epsilon = epsilon
        self.sparsity_coef = sparsity_coef
        self.momentum = momentum
        self.split = split
        self.string_emb_dim = string_emb_dim
        self.num_inputs = [tf.feature_column.numeric_column(x) for x in num_features]
        if string_emb_dim == None:
            self.cat_inputs = [tf.feature_column.indicator_column(
                                tf.feature_column.categorical_column_with_vocabulary_list(x, vocabulary_list=input_df[x].unique())
                                ) for x in cat_features]
            self.dim_imp = len(self.num_inputs) + sum([len(input_df[x].unique()) for x in cat_features])
        else:
            self.cat_inputs = [tf.feature_column.embedding_column(
                                tf.feature_column.categorical_column_with_vocabulary_list(x, vocabulary_list=input_df[x].unique()),
                                dimension=self.string_emb_dim) for x in cat_features]

            self.dim_imp = len(self.num_inputs) + len(self.cat_inputs) * self.string_emb_dim
        self.input_features = tf.keras.layers.DenseFeatures([*self.num_inputs, *self.cat_inputs])
        self.encoder = TabEncoder(dim_imp=self.dim_imp, 
                                  dim_d=self.dim_d, 
                                  dim_a=self.dim_a, 
                                  num_shared_dec_steps=self.num_shared_dec_steps,
                                  num_dec_steps=self.num_dec_steps,
                                  n_steps=self.n_steps, 
                                  gamma=self.gamma, 
                                  epsilon=self.epsilon, 
                                  sparsity_coef=self.sparsity_coef,
                                  momentum = self.momentum,
                                  split=self.split)
        self.final = tf.keras.layers.Dense(units=self.n_out, use_bias=False, activation=self.activation, kernel_initializer=initializer)

        

    def get_config(self):
        config = super(TabNet, self).get_config()
        config['dim_imp'] = self.dim_imp
        config['dim_d'] = self.dim_d
        config['dim_a'] = self.dim_a
        config['num_shared_dec_steps'] = self.num_shared_dec_steps
        config['num_dec_steps'] = self.num_dec_steps
        config['n_steps'] = self.n_steps
        config['n_out'] = self.n_out
        config['activation'] = self.activation
        config['gamma'] = self.gamma
        config['epsilon'] = self.epsilon
        config['sparsity_coef'] = self.sparsity_coef
        config['momentum'] = self.momentum
        config['split'] = self.split
        config['string_emb_dim'] = self.string_emb_dim
        return config

    def call(self, inp, training=None):
        x = self.input_features(inp)
        output, _ = self.encoder(x, training=True)
        output = self.final(output)
        return output

    def explain(self, inp):
        _, importance = self.encoder(inp)
        return importance


    def summary(self, *args, **kwargs):
        super(TabNet, self).summary(*args, **kwargs)
        self.encoder.summary(*args, **kwargs)
 