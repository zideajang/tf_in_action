#coding=utf-8

"""
我们使用 PTB 数据集
包含 10000 不同词语和语句结束
<eos>
"""

import os
import sys
import argparse
import datetime
import collections

import numpy as np
import tensorflow as tf

# 保存训练所得的模型参数文件的目录
data_path = "./data/"
# 保存训练所得的模型参数文件目录
save_path = './save'

load_file = "train-checkpoint-69"

parser = argparse.ArgumentParser()
parser.add_argument('--data_path',type=str,default=data_path,help='The path of the data for training and testing')
args = parser.parse_args()
parser.add_argument('--data_file',type=str,default=data_path,help='The path of the data for training and testing')

# 如果 Python3
Py3 = sys.version_info[0] == 3
# 将文件根据语句结束标识符(<eos>) 来分割
def read_words(filename):
    with tf.io.gfile.GFile(filename,"r") as f:
        if Py3:
            return f.read().replace("\n","<eos>").split()
        else:
            return f.read().decode("utf-8").replace("\n","<eos>").split()

# 构造从单词到唯一整数值的映射
# 单词 the 出现频次最多，对应整数值 0 <unk>
def build_vocab(filename):
    data =  read_words(filename)
    # 将空格间隔的数据转换为键值对
    # Counter({'but': 1, 'while': 1, 'the': 1, 'new': 1, 'york': 1, 'stock': 1})
    
    # 统计单词出现的次数，为了之后按单词次数的多少来排序
    counter = collections.Counter(data)
    # 输出形式为 [('but', 1), ('new', 1), ('stock', 1), ('the', 1), ('while', 1), ('york', 1)]
    count_pairs = sorted(counter.items(),key=lambda x:(-x[1],x[0]))
    # ('but', 'new', 'stock', 'the', 'while', 'york')
    words, _ = list(zip(*count_pairs))

    # 输出数据格式 {'but': 0, 'new': 1, 'stock': 2, 'the': 3, 'while': 4, 'york': 5}
    word_to_id = dict(zip(words,range(len(words))))

    return word_to_id

# 将文件的单词替换为独一的整数
def file_to_word_ids(filename, word_to_id):
    data = read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]

"""
Batch size 批次(目标)数目。一次迭代(Forword)
超参数
init_scale - 权重参数(weights) 的
num_layers - LSTM 层的数目
num_steps - 35 
max_lr_epoch
dropout - 在 Dropout 层的留存率(默认值0.5)
bath_size - 批次(样本)
"""
# 加载所有数据，读取所有单词，把其转换为唯一对应的整数值
def load_data(data_path):
    train_path = os.path.join(data_path,"ptb.train.txt")
    valid_path = os.path.join(data_path,"ptb.valid.txt")
    test_path = os.path.join(data_path,"ptb.test.txt")

    # 建立词汇表，将所有单词(word)转为唯一对应的整数值(id)
    word_to_id = build_vocab(train_path)

    train_data = file_to_word_ids(train_path,word_to_id)
    valid_data = file_to_word_ids(valid_path,word_to_id)
    test_data = file_to_word_ids(test_path,word_to_id)

    # 所有唯一的词汇的个数
    vocab_size = len(word_to_id)

    # 反转一个词汇表：为了之后从整数转为单词
    id_to_word = dict(zip(word_to_id.values(),word_to_id.keys()))

    print(train_data[:10])
    print("===============")
    print(word_to_id)
    print("===============")
    print(vocab_size)
    print("===============")
    print(" ".join([id_to_word[x] for x in train_data[:10]]))
    print("===============")

    return train_data, valid_data, test_data, vocab_size, id_to_word
# 生成批次样本
#
def generate_batches(raw_data, batch_size, num_steps):
    # 将数据转为 Tensor 类型
    raw_data = tf.convert_to_tensor(raw_data,name="raw_data",dtype=tf.int32)
    # 数据集的大小
    data_len = tf.size(raw_data)
    # 批量的大小，// 表示整数除法
    batch_len = data_len // batch_size

    # 原来数据输入形式一维长度为 data_len 的向量，可以通过 batch_size * batch_len 计算得到
    # 通过上面公式可以查看他们之间的关系
    #将数据形状转为 [batch_size, batch_len]
    data = tf.reshape(raw_data[0:batch_size * batch_len],[batch_size,batch_len])

    # 每一次 epoch 数据的大小，因为是从 0 计算所以，在 RNN 中每一次 epoch 输入可以看出一次输入，这是因为这些输入是
    # 相关联的
    epoch_size = (batch_len - 1 )//num_steps
    # range_input_producer 可以用多线程异步的方式从数据里提取数据，用多线程可以加快训练
    # 因为 feed dict 的赋值方式效率不高，大部分分享和资料中都用 feed_dict 来进行赋值 suffle 为 False
    # 表示不打乱数据按照队列先进先出的方式提取数据，也可以尝试 tensorflow 推出最先 api
    i = tf.train.range_input_producer(epoch_size,shuffle=False).dequeue()

    x = data[:,i * num_steps:(i+1) * num_steps]
    x.set_shape([batch_size,num_steps])

    # 
    y = data[:,i * num_steps + 1:(i+1) * num_steps + 1]
    y.set_shape([batch_size,num_steps])

    return x,y # y 是期望
    
# 输入数据
class Input(object):
    def __init__(self, batch_size, num_steps,data):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.epoch_size = ((len(data) // batch_size) -1) // num_steps
        self.input_data, self.targets = generate_batches(data,batch_size,num_steps)
if __name__ == "__main__":
    load_data()