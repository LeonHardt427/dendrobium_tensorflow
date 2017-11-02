#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/1 上午9:38
# @Author  : LeonHardt
# @File    : dendrobium_deep.py

import tensorflow as tf
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
"""
set parameter
"""
INPUT_NODE = 128
OUTPUT_NODE = 10
HIDDEN_LAYER1 = 100
SUMMARY_DIR = '/tmp/to/log'
TEST_DRI = '/tmp/to/test'
TRAIN_STEP = 5000

"""
arrange summaries
"""


def variable_summaries(var, name):
    with tf.name_scope('summaries'):
        tf.summary.histogram(name, var)

        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)

        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev/' + name, stddev)

"""
create one layer with full linked
"""


def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weight'):
            weights = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=0.1))
            variable_summaries(weights, layer_name + '/weights')

        with tf.name_scope('biases'):
            biases = tf.Variable(tf.constant(0.0, shape=[output_dim]))
            variable_summaries(biases, layer_name + '/biases')

        with tf.name_scope('Wx_plus_b'):
            pre_activate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram(layer_name + '/pre_activations', pre_activate)
            activation = act(pre_activate, name='activation')
            tf.summary.histogram(layer_name + '/activation', activation)
            return activation
"""
main_train function
"""


def main(_):
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, None, name='x_input')
        y_ = tf.placeholder(tf.float32, None, name='y_input')

        X_data = np.loadtxt('x_sample.csv', delimiter=',')
        y_data = np.loadtxt('y_label.csv', delimiter=',', dtype='int8')
        y_data = LabelBinarizer().fit_transform(y_data)

        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.1)

        print(y_train.shape)
        print(y_.shape)
        hidden1 = nn_layer(x, 128, 100, 'layer1')
        y = nn_layer(hidden1, 100, 10, 'layer2', act=tf.identity)
        print(y.shape)
# cross_entropy
    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
        tf.summary.scalar('cross entropy', cross_entropy)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        summary_write = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)
        test_write = tf.summary.FileWriter(TEST_DRI, sess.graph)
        tf.global_variables_initializer().run()
        for i in range(TRAIN_STEP):
                summary, _ = sess.run([merged, train_step], feed_dict={x: X_train, y_: y_train})
                test_result = sess.run(merged, feed_dict={x: X_test, y_: y_test})
                summary_write.add_summary(summary, i)
                test_write.add_summary(test_result, i)

    summary_write.close()
    test_write.close()

if __name__ == '__main__':
    tf.app.run()