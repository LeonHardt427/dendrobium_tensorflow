#!/usr/bin/env python
# -*- coding: utf-8 -*
# @Time    : 2017/10/28 上午10:18
# @Author  : LeonHardt
# @File    : dendrobium_eval.py


import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# load the functions and variables in dendrobium_inference and dendrobium_train
import dendrobium_inference
import dendrobium_train


# load the newest model once in 10 seconds, and test the accuracy
EVAL_INTERVAL_SECS = 10


def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, dendrobium_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, dendrobium_inference.OUTPUT_NODE], name='y-input')
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}

# calculate the accuracy from forward propagation
        y = dendrobium_inference.inference(x, None)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# load the model using the method ---rename
        variable_average = tf.train.ExponentialMovingAverage(dendrobium_train.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_average.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(dendrobium_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
# get the time from document name
                    global_step = ckpt.model_checkpoint_path.splite('/')[-1].splite('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print("After %s training step(s), validation accuracy = %g" % (global_step, accuracy_score))
                else:
                    print("No checkpoint dile found")
                    return
                time.sleep(EVAL_INTERVAL_SECS)

def main(argv=None):
    mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
    evaluate(mnist)

if __name__ == '__main__':
    tf.app.run()
