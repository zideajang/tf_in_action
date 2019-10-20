# coding=utf-8
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

const1 = tf.constant([[2,2]])
const2 = tf.constant([[4],[4]])

print(const1)
print(const2)

multiple = tf.matmul(const1,const2)

print(multiple)

sess = tf.Session()

result = sess.run(multiple)

print(result)
