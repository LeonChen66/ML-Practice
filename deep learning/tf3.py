# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 20:07:29 2017

@author: Leon
"""

import tensorflow as tf

state = tf.Variable(0, name='counter')
#print(state.name)
one = tf.constant(1)

new_value = tf.add(state, one)
update = tf.assign(state, new_value)

#must have if define variable
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))