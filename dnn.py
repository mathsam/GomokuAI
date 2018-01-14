import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from dnn_diagnostic import count_likely_moves

input_height = 9
input_width = 9
input_channels = 1
conv_n_maps = [64, 64, 64]
conv_kernel_sizes = [(3,3), (3,3), (3,3)]
conv_strides = [1, 1, 1]
conv_paddings = ["SAME"] * 3
conv_activation = [tf.nn.relu] * 3
l2_regu_coef = 2e-2
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

learning_rate = 1e-4
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


def softmax(array_input):
    exp_array = np.exp(array_input)
    return exp_array/np.sum(exp_array, axis=-1)[...,np.newaxis]


class AINet(object):

    chkpoint_name = 'policy-value-network-chk'

    def __init__(self, start_type='restart', load_path=r'./dnn_data/v0', save_path=None, use_gpu=True):
        """
        :param start_type: 'new'|'restart'
        """
        session_conf = None
        if not use_gpu:
            session_conf = tf.ConfigProto(
                device_count={'CPU' : 1, 'GPU' : 0})

        self.tf_session = tf.Session(config=session_conf)
        self.saver = tf.train.Saver()
        self.load_path = load_path
        if save_path is None:
            self.save_path = load_path
        else:
            self.save_path = save_path
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)

        if start_type == 'new':
            init = tf.global_variables_initializer()
            init.run(session=self.tf_session)
        else:
            self.saver.restore(self.tf_session, os.path.join(self.load_path, self.chkpoint_name))

    def save(self):
        saved_dir = self.saver.save(self.tf_session, os.path.join(self.save_path, self.chkpoint_name))
        print('State saved at: %s' %saved_dir)

    def train(self, board_sample, move_sample, result_sample):
        batch_size = 1024
        for i in range(int(1e5)):
            offset = (i * batch_size) % (board_sample.shape[0] - batch_size)
            board_batch = board_sample[offset:(offset + batch_size),...]
            move_batch = move_sample[offset:(offset + batch_size),...]
            result_batch = result_sample[offset:(offset + batch_size),...]
            _, loss_val = self.tf_session.run([training_op, loss], feed_dict={
                board_state: board_batch, train_move: move_batch, train_value: result_batch})
            if i % 100 == 0:
                print('step %d: loss %g' %(i, loss_val))
            if i % 1e4 == 0:
                self.save()
                print('\nsaved at step %d\n' %i)

    def pred(self, curr_board):
        p_and_val = policy_and_value.eval(feed_dict={board_state: [curr_board]}, session=self.tf_session).flatten()
        p = softmax(p_and_val[:-1])
        val = p_and_val[-1]
        return p, val

    def pred_batch(self, board_states):
        p_and_vals = policy_and_value.eval(feed_dict={board_state: board_states}, session=self.tf_session)
        return p_and_vals

    def eval_loss(self, board_sample, move_sample, result_sample):
        loss_val = loss.eval(feed_dict={board_state: board_sample,
                                        train_move: move_sample,
                                        train_value: result_sample},
                             session=self.tf_session)
        return loss_val


def produce_test_stats(X_test, Y_test):
    import scipy
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.values
    if isinstance(Y_test, pd.DataFrame):
        Y_test = Y_test.values

    board_sample_test = X_test.astype(np.float32)
    board_sample_test = np.reshape(board_sample_test, [-1, input_height, input_width, 1])
    move_sample_test = Y_test[:,:input_height*input_width].astype(np.float32)
    result_sample_test = Y_test[:,-1].astype(np.float32)

    prediction = ai_net.pred_batch(board_sample_test)
    test_loss = ai_net.eval_loss(board_sample_test, move_sample_test, result_sample_test)
    print('Test loss: %f' %test_loss)

    value_prediction = prediction[:, -1]
    value_test = Y_test[:, -1]

    value_precision = (np.sign(value_prediction) == np.sign(value_test)).sum() / float(len(value_test))
    print('Value precision: %f' %value_precision)

    move_prob = softmax(prediction[:, :-1])
    pred_move = np.argmax(prediction[:, :-1], axis=1)
    actual_move = np.argmax(Y_test[:, :-2], axis=1)

    move_precision = (pred_move == actual_move).sum() / float(len(pred_move))
    print('Move precision: %f' %move_precision)

    top_k_move_precision = count_likely_moves(Y_test[:, :-2], move_prob)
    print("Top 5 move precision: %f" %top_k_move_precision)

    plt.figure()
    plt.hist(value_prediction[value_test>0], normed=True, alpha=0.5)
    plt.hist(value_prediction[value_test<0], normed=True, alpha=0.5)

    plt.figure()
    plt.plot(move_prob[:100:10,:].T, 'r')
    plt.plot(Y_test[:100:10,:-2].T, 'b')

    plt.figure()
    plt.scatter(scipy.special.logit(move_prob.flatten()), scipy.special.logit(Y_test[:,:-2].flatten()))



if __name__ == '__main__':
    import pandas as pd
    import time
    print '\n---------------------Training Start-----------------------'
    print 'Training start at ', time.ctime()
    start_time = time.time()

    base_dir = r'./dnn_data'
    training_status = eval(open(os.path.join(base_dir, 'training_status')).read())
    save_path = os.path.join(base_dir, 'v%d' %training_status['current_challenger'])
    if (not os.path.isdir(save_path)) or (not os.listdir(save_path)):
        load_path = os.path.join(base_dir, 'v%d' %training_status['current_champion'])
    else:
        load_path = save_path
    print 'Load existing model from ', load_path
    print 'Model will be saved at ', save_path

    ai_net = AINet('restart', load_path=load_path, save_path=save_path)

    X_train = pd.read_csv(os.path.join('analysis', 'X_train.csv'))
    Y_train = pd.read_csv(os.path.join('analysis', 'Y_train.csv'))

    board_sample = X_train.values.astype(np.float32)
    board_sample = np.reshape(board_sample, [-1, input_height, input_width, 1])
    move_sample = Y_train.iloc[:, :input_height*input_width].values.astype(np.float32)
    result_sample = Y_train.loc[:, 'value'].values.astype(np.float32)

    ai_net.train(board_sample, move_sample, result_sample)
    ai_net.save()

    #produce_test_stats(X_train.loc[:5000,:], Y_train.loc[:5000,:])
    #X_test = pd.read_csv(os.path.join('analysis', 'X_test.csv'))
    #Y_test = pd.read_csv(os.path.join('analysis', 'Y_test.csv'))
    #produce_test_stats(X_test, Y_test)

    print 'Training end at ', time.ctime()
    end_time = time.time()
    print 'Time consumed for training ', end_time - start_time
    print '---------------------Training End-----------------------\n'

