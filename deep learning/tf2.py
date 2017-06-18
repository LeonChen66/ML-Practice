# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 20:01:29 2017

@author: Leon
"""

import tensorflow as tf
import  numpy as np

matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],
                       [2]])

product = tf.matmul(matrix1, matrix2)    #matrix multiply

#method 1
"""
sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()
"""

#method 2
with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)
    