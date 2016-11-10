import numpy as np
import tensorflow as tf
import scipy.sparse as sp


def simple_batcher(x, y, bs):
    for begin, end in zip(range(0, len(x), bs)[:-1], range(0, len(x), bs)[1:]):
        yield (x[begin:end], y[begin:end])


class TFExpMachine:
    def __init__(self, rank=5, s_features=[2, 3, 4, 5], init_std=0.7, exp_reg=1.1, reg=0.01, seed=42):
        self.rank =  rank
        self.s_features = s_features
        self.n_features = len(s_features)
        self.init_std = init_std
        self.graph = tf.Graph()
        if seed:
            self.graph.seed = seed
        self.exp_reg = exp_reg
        self.reg = reg
        self_init_cores = None
        
    def init_from_cores(self, core_list):
        assert(len(core_list) == len(self.s_features))
        self.init_vals = core_list
        
    
    def build_graph(self):
        with self.graph.as_default():
            
            # placeholders
            self.X = tf.placeholder(tf.int64, [None, self.n_features], name='X')
            self.Y = tf.placeholder(tf.float32, (None), name='Y')
            
            # list of TT-cores
            self.G = [None]*self.n_features

            # list of TT-cores used for penalty
            self.G_exp = [None]*self.n_features
            
            for i in range(self.n_features):

                shape = [self.s_features[i] + 1, self.rank, self.rank]
                if i==0:
                    shape = [self.s_features[i] + 1, 1, self.rank]
                if i==(self.n_features - 1):
                    shape = [self.s_features[i] + 1, self.rank, 1]

                content = None
                if self.init_vals is None:
                    content = tf.random_normal(shape, stddev=self.init_std)
                else:
                    assert(self.init_vals[i].shape==tuple(shape))
                    content = self.init_vals[i] + tf.random_normal(shape, stddev=self.init_std)

                self.G[i] = tf.Variable(content, trainable=True, name='G_{}'.format(i))
                exp_weights = tf.constant([1] + [self.exp_reg] * self.s_features[i], shape=(self.s_features[i] + 1, 1, 1))
                self.G_exp[i] = self.G[i] * exp_weights

            # main computation part
            cur_col = self.X[:, 0]
            tower = tf.gather(self.G[0], cur_col)
            self.outputs = tf.add(self.G[0][0], tower)
            for i in range(1, self.n_features):
                cur_col = self.X[:, i]
                cur_tower = tf.gather(self.G[i], cur_col)
                cur_A = tf.add(self.G[i][0], cur_tower)
                self.outputs = tf.batch_matmul(self.outputs, cur_A)
            self.outputs = tf.squeeze(self.outputs, [1, 2])
            
            # regularization penalty
            self.penalty = tf.reshape(
                tensor=tf.einsum('nip,njq->ijpq', self.G_exp[0], self.G_exp[0]),
                shape=(1, self.rank**2)
            )
            for i in range(1, self.n_features):
                last_dim = 1 if i==self.n_features-1 else self.rank**2
                summed_kron_prod = tf.reshape(
                    tensor=tf.einsum('nip,njq->ijpq', self.G_exp[i], self.G_exp[i]),
                    shape=(self.rank**2, last_dim)
                )
                self.penalty = tf.matmul(self.penalty, summed_kron_prod)

            # MSE loss
            self.loss = tf.reduce_mean((self.outputs - self.Y)**2)
            # # LogLoss
            # self.margins = -self.Y * self.outputs
            # sself.raw_loss = tf.log(tf.add(1.0, tf.exp(self.margins)))
            # self.loss = tf.reduce_mean(tf.minimum(self.raw_loss, 100, name='truncated_log_loss'))
            self.penalized_loss = self.loss + self.reg * tf.squeeze(self.penalty)

            # others
            self.trainer = tf.train.AdamOptimizer(0.001).minimize(self.penalized_loss)
            self.init_all_vars = tf.initialize_all_variables()
            self.saver = tf.train.Saver()

    def initialize_session(self):
        config = tf.ConfigProto()
        # for reduce memory allocation
        config.gpu_options.allow_growth = True
        self.session = tf.Session(graph=self.graph, config=config)
        self.session.run(self.init_all_vars)

    def destroy(self):
        self.session.close()
        self.graph = None