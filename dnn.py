import numpy as np
import tensorflow as tf

input_height = 9
input_width = 9
input_channels = 1
conv_n_maps = [64, 64, 64]
conv_kernel_sizes = [(3,3), (3,3), (3,3)]
conv_strides = [1, 1, 1]
conv_paddings = ["SAME"] * 3
conv_activation = [tf.nn.relu] * 3
l2_regu_coef = 1e-4
hidden_activation = tf.nn.relu
n_outputs = input_height * input_width
initializer = tf.contrib.layers.variance_scaling_initializer()
l2_regularizer = tf.contrib.layers.l2_regularizer(scale=l2_regu_coef)


def q_network(X_state, name):
    prev_layer = X_state
    with tf.variable_scope(name) as scope:
        for n_maps, kernel_size, strides, padding, activation in zip(
                conv_n_maps, conv_kernel_sizes, conv_strides,
                conv_paddings, conv_activation):
            prev_layer = tf.layers.conv2d(
                prev_layer, filters=n_maps, kernel_size=kernel_size,
                strides=strides, padding=padding, activation=activation,
                kernel_initializer=initializer, kernel_regularizer=l2_regularizer)
        policy_head = tf.layers.conv2d(prev_layer, filters=2, kernel_size=(1,1), strides=1, padding='SAME',
                                       activation=tf.nn.relu, kernel_initializer=initializer,
                                       kernel_regularizer=l2_regularizer)
        policy_head_flat = tf.reshape(policy_head, shape=[-1, 2*input_height*input_width])
        policy_output = tf.layers.dense(policy_head_flat, input_height*input_width, activation=None,
                                        kernel_initializer=initializer,
                                        kernel_regularizer=l2_regularizer)

        value_head = tf.layers.conv2d(prev_layer, filters=1, kernel_size=(1,1), strides=1, padding='SAME',
                                      activation=tf.nn.relu, kernel_initializer=initializer,
                                      kernel_regularizer=l2_regularizer)
        value_head_flat = tf.reshape(value_head, shape=[-1, input_height*input_width])
        value_output = tf.layers.dense(value_head_flat, 1, activation=tf.nn.tanh, kernel_initializer=initializer,
                                       kernel_regularizer=l2_regularizer)

    policy_and_value = tf.concat([policy_output, value_output], axis=1)

    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope=scope.name)
    trainable_vars_by_name = {var.name[len(scope.name):]: var
                              for var in trainable_vars}
    return policy_and_value, trainable_vars_by_name


board_state = tf.placeholder(tf.float32, shape=[None, input_height, input_width,
                                                input_channels])

policy_and_value, network_params = q_network(board_state, name='value.policy.network')

learning_rate = 0.001
momentum = 0.9

with tf.variable_scope("train"):
    train_move = tf.placeholder(tf.float32, shape=[None, input_height*input_width])
    train_value = tf.placeholder(tf.float32, shape=[None])
    pred_policy = policy_and_value[:,:-1]
    pred_value = policy_and_value[:,-1]
    loss = (tf.losses.softmax_cross_entropy(train_move, pred_policy) +
            tf.losses.mean_squared_error(train_value, pred_value))
    reg_vars = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'value.policy.network')
    reg_term = tf.contrib.layers.apply_regularization(l2_regularizer, reg_vars)

    loss += reg_term

    global_step = tf.Variable(0, trainable=False, name='global_step')
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True)
    training_op = optimizer.minimize(loss, global_step=global_step)


def softmax(array1d):
    exp_array = np.exp(array1d)
    return exp_array/np.sum(exp_array)

class AINet(object):
    checkpoint_path = './policy-value-network-chk'

    def __init__(self, start_type='restart'):
        """
        :param start_type: 'new'|'restart'
        """
        self.tf_session = tf.Session()
        self.saver = tf.train.Saver()
        if start_type == 'new':
            init = tf.global_variables_initializer()
            init.run(session=self.tf_session)
        else:
            self.saver.restore(self.tf_session, self.checkpoint_path)

    def save(self):
        self.saver.save(self.tf_session, self.checkpoint_path)

    def train(self, board_sample, move_sample, result_sample):
        batch_size = 256
        for i in range(int(1e6)):
            offset = (i * batch_size) % (board_sample.shape[0] - batch_size)
            board_batch = board_sample[offset:(offset + batch_size),...]
            move_batch = move_sample[offset:(offset + batch_size),...]
            result_batch = result_sample[offset:(offset + batch_size),...]
            _, loss_val = self.tf_session.run([training_op, loss], feed_dict={
                board_state: board_batch, train_move: move_batch, train_value: result_batch})
            if i % 100 == 0:
                print('step %d: loss %g' %(i, loss_val))

    def pred(self, curr_board):
        p_and_val = policy_and_value.eval(feed_dict={board_state: [curr_board]}, session=self.tf_session).flatten()
        p = softmax(p_and_val[:-1])
        val = p_and_val[-1]
        return p, val


if __name__ == '__main__':
    import os
    import pandas as pd
    ai_net = AINet('restart')
    X_train = pd.read_csv(os.path.join('analysis', 'X_train.csv'))
    Y_train = pd.read_csv(os.path.join('analysis', 'Y_train.csv'))

    board_sample = X_train.values.astype(np.float32)
    board_sample = np.reshape(board_sample, [-1, input_height, input_width, 1])
    move_sample = Y_train.iloc[:, :input_height*input_width].values.astype(np.float32)
    result_sample = Y_train.loc[:, 'value'].values.astype(np.float32)

    ai_net.train(board_sample, move_sample, result_sample)
    ai_net.save()
