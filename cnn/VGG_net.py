import tensorflow as tf
# print(tf.__version__)
import numpy as np
import cPickle
import os
import utils
from utils import CifarData
# save file format as numpy
# file format as cPickle
CIFAR_DIR = "./cifar-10-batches"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# print(os.listdir(CIFAR_DIR))

train_filenames = [os.path.join(CIFAR_DIR, 'data_batch_%d' % i)
                   for i in range(1, 6)]
test_filenames = [os.path.join(CIFAR_DIR, 'test_batch')]
train_data = CifarData(train_filenames, True)
test_data = CifarData(test_filenames, False)

# None present confirm count of smaples
x = tf.placeholder(tf.float32,[None,3072])
y = tf.placeholder(tf.int64,[None])

x_image = tf.reshape(x,[-1,3,32,32])
# 32 * 32
x_image = tf.transpose(x_image,perm=[0,2,3,1])

# neru feature_map image output
conv1_1 = tf.layers.conv2d(x_image,32,(3,3),padding='same',activation=tf.nn.relu,name='conv1_1')
conv1_2 = tf.layers.conv2d(conv1_1,32,(3,3),padding='same',activation=tf.nn.relu,name='conv1_2')
# 16 * 16
pooling1 = tf.layers.max_pooling2d(conv1_2,(2,2),(2,2),name='pool1')

conv2_1 = tf.layers.conv2d(pooling1,32,(3,3),padding='same',activation=tf.nn.relu,name='conv2_1')
conv2_2 = tf.layers.conv2d(conv2_1,32,(3,3),padding='same',activation=tf.nn.relu,name='conv2_2')
# 8 * 8
pooling2 = tf.layers.max_pooling2d(conv2_2,(2,2),(2,2),name='pool2')

conv3_1 = tf.layers.conv2d(pooling2,32,(3,3),padding='same',activation=tf.nn.relu,name='conv3_1')
conv3_2 = tf.layers.conv2d(conv3_1,32,(3,3),padding='same',activation=tf.nn.relu,name='conv3_2')
# 4 * 4
pooling3 = tf.layers.max_pooling2d(conv3_2,(2,2),(2,2),name='pool3')

# [None , 4 * 4 * 32]
flatten = tf.layers.flatten(pooling3)
y_ = tf.layers.dense(flatten,10)
"""
hidden1 = tf.layers.dense(x,100,activation=tf.nn.relu)
hidden2 = tf.layers.dense(hidden1,100,activation=tf.nn.relu)
hidden3 = tf.layers.dense(hidden2,50,activation=tf.nn.relu)
[Train] Step: 9999, loss: 1.24794, acc: 0.45000
[Test ] Step: 10000, acc: 0.50300
"""
# simplify with tensorflow api
# y_ = tf.layers.dense(hidden3,10)

loss = tf.losses.sparse_softmax_cross_entropy(labels=y,logits=y_)

# indices
predict = tf.argmax(y_,1)
# equal 
correct_prediction = tf.equal(predict,y)

# [1,0,0,1,1,]
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float64))

train_data = CifarData(train_filenames,True)
# test_data = CifarData(test_filenames,False)
batch_size = 20
train_steps = 10000
test_steps = 100
with tf.name_scope('train_op'):
    # learning rate le-3 
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(train_steps):
        batch_data,batch_labels = train_data.next_batch(batch_size)
        loss_val, accu_val,_ = sess.run([loss,accuracy,train_op],feed_dict={x: batch_data,y: batch_labels})
        if (i+1) % 500 == 0:
            print '[Train] Step: %d, loss: %4.5f, acc: %4.5f' % (i, loss_val, accu_val)
        if (i+1) % 1000 == 0:
            test_data  = CifarData(test_filenames,False)
            all_test_acc_val = []
            for j in range(test_steps):
                test_batch_data,test_batch_labels = test_data.next_batch(batch_size)
                test_acc_val = sess.run([accuracy],
                    feed_dict={x: test_batch_data,y: test_batch_labels })
                all_test_acc_val.append(test_acc_val)
            test_acc = np.mean(all_test_acc_val)
            print '[Test ] Step: %d, acc: %4.5f ' % (i+1,test_acc)