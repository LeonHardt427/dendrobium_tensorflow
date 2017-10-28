#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/27 下午4:28
# @Author  : LeonHardt
# @File    : dendrobium_train.py

import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import  dendrobium_inference


# Set parameters
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 3
MOVING_AVERAGE_DECAY = 0.99
# Saving path
MODEL_SAVE_PATH = "./path/to/model"
MODEL_NAME = "model.ckpt"


def train(mnist):
    x = tf.placeholder(tf.float32, [None, dendrobium_inference.INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, dendrobium_inference.OUTPUT_NODE], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
# using forward propagation in 'dendrobium_inference'
    y = dendrobium_inference.inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(y, tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE, global_step, mnist.train.num_example / BATCH_SIZE, LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

# initialize forever tensorflow class
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss,global_step], feed_dict={x: xs, y_: ys})

# save the result once in 1000 times
            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

def main(argv = None):
    mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()






















