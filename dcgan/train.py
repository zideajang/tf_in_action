# coding=utf-8
"""
训练 DCGAN
"""

import glob
import numpy as np
from scipy import misc
import tensorflow as tf

from network import *

def train():
    # 获取训练数据 images 获取训练数据
    data = []
    for image in glob.glob("images/*"):
        image_data = misc.imread(image) # imread 利用 PIL 来读取图片数据

        data.append(image_data)
    input_data = np.array(data)
    # 将数据标准化成 [-1,1] 的取值,这也是 Tanh 激活函数的输出范围
    input_data = (input_data.astype(np.float32) -  127.5) / 127.5

    # 将数据标准化，构造生成器和判别器
    g = generator_model()
    d = discriminator_model()

    # 构建生成器和判别器组成的网络模型
    d_on_g = generator_containing_discriminator(g,d)

    # 优化器用 Adam Optimizer
    g_optimizer = tf.keras.optimizers.Adam(lr=LEARNING_RATE,beta_1=BETA_1)
    d_optimizer = tf.keras.optimizers.Adam(lr=LEARNING_RATE,beta_1=BETA_1)

    # Sequential.compile 对神经网进行一些配置
    # 配置生成器和判别器
    g.compile(loss="binary_crossentropy",optimizer=g_optimizer)
    d.compile(loss="binary_crossentropy",optimizer=d_optimizer)
    d.trainable = True
    # 用于生成器
    d_on_g.compile(loss="binary_crossentropy",optimizer=g_optimizer)

    for epoch in range(EPOCHS):
        for index in range(int(input_data.shape[0]/BATCH_SIZE)):
            input_batch = input_data[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
            # 连续型均匀
if __name__ == "__main__":
    train()
