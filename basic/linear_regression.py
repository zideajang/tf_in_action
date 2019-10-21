#coding=utf-8
"""
线性回归
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# 构造数据
points_num = 100
vectors = []
# 用 numpy 的正态随机分布函数生成 100 个点，这些点(x,y)坐标值x 和 y 关系对应线性方程 y = 0.1 * x + 0.5
# 权重为 0.1 偏差为 0.5

for i in range(points_num):
    x1 = np.random.normal(0.0, 0.66)
    y1 = 0.1 * x1 + 0.5 + np.random.normal(0.0,0.04)
    vectors.append([x1, y1])

x_data = [v[0] for v in vectors] # x 坐标(输入值)
y_data = [v[1] for v in vectors] # y 坐标(预期值)

# 图像 1：展示 100 随机数据点
# plt.plot(x_data,y_data,'b*',label="Original data")
# plt.show()

# 用梯度下降来解决这个问题
# 构建线性回归模型
W = tf.Variable(tf.random_uniform([1],-1.0,1.0)) # 初始化权重
b = tf.Variable(tf.zeros([1])) # 初始化 Bias
y = W * x_data + b #模型计算出来 y

# 定义 loss function(损失函数)
# 对 Tensor 的所有维度计算 $(y - y_data)^2/points_num $
#
loss = tf.reduce_mean(tf.square(y - y_data))

# 用梯度下降优化器来优化 loss 函数
optimizer = tf.train.GradientDescentOptimizer(0.5) #设置学习率为 0.5 
train = optimizer.minimize(loss)

# 创建会话
sess = tf.Session()
# 初始化数据数据流图中所有变量
init = tf.global_variables_initializer()
sess.run(init)

# 训练 20 步
steps = 20
for step in range(steps):
    # 优化每一步
    sess.run(train)
    # 打印出每一步的损失、权重和偏差
    print("step=%d loss=%f, [Weight=%f,Bias=%f]" % (step,sess.run(loss),sess.run(W),sess.run(b)))
    
# 图像绘制所有点并且绘制出得到最佳拟合的曲线(直线)

plt.plot(x_data,y_data,'b*',label="Original data")
plt.title("Linear Regression using Gradient Descent")
plt.plot(x_data,sess.run(W) * x_data + sess.run(b), label="Fitted line")
plt.legend()
plt.show()

# 关闭会话
sess.close()