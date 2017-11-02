#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/31 上午9:20
# @Author  : LeonHardt
# @File    : dendrobium_deep_simple.py

import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer

#读数据
digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


def add_layer(inputs, in_size, out_size, layer_name, activation_function=None):   # activation_function=None线性函数
        Weights = tf.Variable(tf.random_normal([in_size, out_size]))           # Weight中都是随机变量
        biases = tf.Variable(tf.zeros([1, out_size])+0.1)                      # biases推荐初始值不为0
        Wx_plus_b = tf.matmul(inputs, Weights)+biases                          # inputs*Weight+biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        tf.histogram_summary(layer_name+'/outputs', outputs)         # histogram_summary和scalar_summary对应
        return outputs

xs = tf.placeholder(tf.float32, [None, 64])           # 8*8，64个单位
ys = tf.placeholder(tf.float32, [None, 10])

l1 = add_layer(xs, 64, 50, 'l1', activation_function=tf.nn.tanh)        # 隐藏层（输出100为了展示过拟合）
prediction = add_layer(l1, 50, 10, 'l2', activation_function=tf.nn.softmax)      # 输出层

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction), reduction_indices=[1]))
tf.scalar_summary('loss', cross_entropy)             # 可视化记录

train_step = tf.train.GradientDescentOptimizer(0.6).minimize(cross_entropy)

init = tf.initialize_all_variables()
sess = tf.Session()
merged = tf.merge_all_summaries()
train_write = tf.train.SummaryWriter("/logs/train", sess.graph)
test_write = tf.train.SummaryWriter("/logs/test", sess.graph)

sess.run(init)

for i in range(1000):
        sess.run(train_step, feed_dict={xs: X_train, ys: y_train})
        if i % 50 == 0:
            train_result = sess.run(merged, feed_dict={xs: X_train, ys: y_train})
            test_result = sess.run(merged, feed_dict={xs: X_test, ys: y_test})
            train_write.add_summary(train_result, i)
            test_write.add_summary(test_result, i)
