import tensorflow as tf


class DQN:
    def __init__(self, params, name, bootstrap_index, last_layer_out, rews, terms, q_t, acts):
        self.params = params
        self.network_name = name

        # flat
        o3_shape = last_layer_out.get_shape().as_list()

        # fc3
        layer_name = 'fc4'
        hiddens = 512
        dim = o3_shape[1] * o3_shape[2] * o3_shape[3]
        self.o3_flat = tf.reshape(last_layer_out, [-1, dim], name=self.network_name + '_' + layer_name + '_input_flat_' + str(bootstrap_index))
        self.w4 = tf.Variable(tf.random_normal([dim, hiddens], stddev=0.01), name=self.network_name + '_' + layer_name + '_weights_' + str(bootstrap_index))
        self.b4 = tf.Variable(tf.constant(0.1, shape=[hiddens]), name=self.network_name + '_' + layer_name + '_biases_' + str(bootstrap_index))
        self.ip4 = tf.add(tf.matmul(self.o3_flat, self.w4), self.b4, name=self.network_name + '_' + layer_name + '_ips_' + str(bootstrap_index))
        self.o4 = tf.nn.relu(self.ip4, name=self.network_name + '_' + layer_name + '_activations_' + str(bootstrap_index))

        # fc4
        layer_name = 'fc5'
        hiddens = params['num_act']
        dim = 512
        self.w5 = tf.Variable(tf.random_normal([dim, hiddens], stddev=0.01), name=self.network_name + '_' + layer_name + '_weights_' + str(bootstrap_index))
        self.b5 = tf.Variable(tf.constant(0.1, shape=[hiddens]), name=self.network_name + '_' + layer_name + '_biases_' + str(bootstrap_index))
        self.y = tf.add(tf.matmul(self.o4, self.w5), self.b5, name=self.network_name + '_' + layer_name + '_outputs_' + str(bootstrap_index))

        if name == 'qnet':
            # Q,Cost,Optimizer
            self.discount = tf.constant(self.params['discount'])
            self.yj = tf.add(rews, tf.multiply(1.0 - terms, tf.multiply(self.discount, q_t)))
            self.Qxa = tf.multiply(self.y, acts)
            self.Q_pred = tf.reduce_max(self.Qxa, reduction_indices=1)
            self.diff = tf.subtract(self.yj, self.Q_pred)

            if self.params['clip_delta'] > 0:
                self.quadratic_part = tf.minimum(tf.abs(self.diff), tf.constant(self.params['clip_delta']))
                self.linear_part = tf.subtract(tf.abs(self.diff), self.quadratic_part)
                self.diff_square = 0.5 * tf.pow(self.quadratic_part, 2) + self.params['clip_delta'] * self.linear_part
            else:
                self.diff_square = tf.multiply(tf.constant(0.5), tf.pow(self.diff, 2))

            if self.params['batch_accumulator'] == 'sum':
                self.loss = tf.reduce_sum(self.diff_square, name = 'cost_' + str(bootstrap_index))  # output
            else:
                self.loss = tf.reduce_mean(self.diff_square, name = 'cost_' + str(bootstrap_index))  # output

            self.rmsprop = tf.train.RMSPropOptimizer(name = 'rmsp_' + str(bootstrap_index), learning_rate = self.params['lr'], decay = self.params['rms_decay'], momentum = 0.95, epsilon = self.params['rms_eps']).minimize(self.loss)