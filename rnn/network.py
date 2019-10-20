#coding=utf8
"""
神经网模型
RNN-LSTM 循环神经网络
输入的维度 (20 35 650)
LSTM 层:神经元数目 650
LSTM 层:神经元数目 650
LSTM 输出维度(20,35,650)
softmax 的参数维度(650 10000)
softmax 的输出维度(700 10000)
"""

import tensorflow as tf
# 神经网络的模型
class Model(object):
    # 构造函数
    # is_training 用于控制是测试还是训练
    # hidden_size LSTM 神经元数量
    # vocab_size 这里词量 10000
    def __init__(self,input, is_training,hidden_size,vocab_size,num_layers,dropout=0.5,init_scale=0.05):
        self.is_training = is_training
        self.input_obj = input #在 utils 定义好的 Input 类的实例
        self.batch_size = input.batch_size
        self.num_steps = input.num_steps #是 STML 在时间维度上展开序列的大小
        self.hidden_size = hidden_size

        # 指定 cpu 来进行运算，我的 mac 还有没有 GPU
        with tf.device("/cpu:0"):
            # 创建词向量(word Embedding), Embedding 表示 Dense Vector(密集向量)
            # 词向量本质上是一种单词聚类(clustering)的方法
            # init_scale 跨度

            embedding = tf.Variable(tf.random_uniform([vocab_size,self.hidden_size],-init_scale,init_scale))
            # embedding_lookup 返回词向量
            # 完成从 vocab_size(10000) 到 hidden_size(650) 的词向量映射
            inputs = tf.nn.embedding_lookup(embedding,self.input_obj.input_data)
            # 词向量本质上是

            # 在训练时，如果 dropout 率小于 1 使输入经过一个 Dropput，Dropout 作用是防止过拟合
            if is_training and dropout < 1:
                inputs = tf.nn.dropout(inputs,dropout)
            # 状态(state)存储和提取
            # 第二维是 2 LSTM 单元有两个来自上一个单元的输入 $C_{t-1} $ 和 $h_{t-1}$
            # $C_{t-1} $ 是前一刻的单元状态
            # $h_{t-1} $ 是前一时刻 LSTM 的输出
            # 这个 c 和 h 用于构建之后的 tf.contrib.rnn.LSTMStateTuple
            self.init_state = tf.placeholder(tf.float32,[num_layers,2,self.batch_size,self.hidden_size])

            # 每一层的状态
            state_per_layer_list = tf.unstack(self.init_state,axis=0)

            # 初始的状态（包含前一时刻 LSTM 的输出 $h_{t-1}$ 和前一时刻的单元状态$c_{t-1}$ 用于之后的 dynamic_rnn
            # 第一次做得到这里也是一脸迷茫，不知所措只好硬着头皮跟着 coding 下去
            rnn_tuple_state = tuple([tf.contrib.rnn.LSTMStateTuple(state_per_layer_list[idx][0],state_per_layer_list[idx][1]) for idx in range(num_layers)])
            # 创建一个 LSTM 层，其中神经元数目是 hidden_size 默认是 650
            cell = tf.contrib.rnn.LSTMCell(hidden_size)
            # num_units 神经元数目

            if is_training and dropout < 1:
                cell = tf.contrib.rnn.DropoutWrapper(cell,output_keep_prob=dropout)
            
            # 如果 LSTM 的层数大于 1 则总计创建 num_layers 个 LSTM 层，将所有的 LSTM 层包装进 MultiRNNCell 这样
            # 序列化层级模型中 state_is_tuple=True 表示接受 LSTMStateTuple 形式输入状态

            if num_layers > 1:
                cell = tf.contrib.rnn.MultiRNNCell([cell for _ in range(num_layers)],state_is_tuple=True)

            # dynamic_rnn(动态 RNN) 可以让不同迭代传入的 Batch 可以是长度不同的数据
            """
            但同一次迭代中一个 Batch 内部的所有数据长度依然是固定的，dynamic_rnn 能够更好处理 padding(补零)的情况，节约
            计算资源，当然除了 dynamic_rnn 还有 rnn 方法，这里因为 dynamic_rnn 优于 rnn 所以使用 dynamic_rnn。
            有两个返回值
            - Batch 里在时间维度(默认是 35)上展开的所有 LSTM 单元的输出，形状默认为 [20,35,650],之后经过偏平层处理
            - 最终 state(状态)，包含当前时刻 LSTM 的输出 $h_t$ 和当前时刻的单元状态$C_t$
            """
            output, self.state = tf.nn.dynamic_rnn(cell,inputs,dtype=tf.float32,initial_state=rnn_tuple_state)
            # 扁平化处理，改变输出形状为 batch_size * num_steps , hidden_size 默认形状为 [700,650]
            output = tf.reshape(output,[-1,hidden_size]) # -1 表示自动推导维度大小
            # softmax 的权重 (weight)
            softmax_w = tf.Variable(tf.random_uniform([hidden_size,vocab_size],-init_scale,init_scale))
            # Softmax 的偏置 (Bias)
            softmax_b = tf.Variable(tf.random_uniform([vocab_size],-init_scale,init_scale))

            # logits 是 Logistic Regression(用于分类)模型(线性方程: y = W * x + b) 计算的结果(分值)
            """
            这个 logits(分值)之后会用 Softamx 来转成百分比概率，output 是输入 (x), softmax_w 是权重(W),softmax_b 是
            偏置(b)返回 W*x + b 结果
            """
            logits = tf.nn.xw_plus_b(output,softmax_w,softmax_b)

            # 将 logits 转化为三维的 Tensor，为了 sequence loss 的计算形状为 [20,35,10000]
            logits = tf.reshape(logits,[self.batch_size,self.num_steps,vocab_size])

            # 计算 logits 的序列的交叉熵(Cross-entropy) 的损失(loss)
            loss = tf.contrib.seq2seq.sequence_loss(
                    logits, # 默认形状 [20,35,10000]
                    self.input_obj.targets, # 期望输出 形状为 [20,35]
                    tf.ones([self.batch_size,self.num_steps],dtype=tf.float32),
                    average_across_timesteps=False,
                    average_across_batch=True)

            # 更新代价(cost)
            self.cost = tf.reduce_sum(loss)

            # softmax 计算出的概率 [700,10000]
            self.softmax_out = tf.nn.softmax(tf.reshape(logits,[-1,vocab_size]))

            # 最最大概率的那个值作为预测
            self.predict = tf.cast(tf.argmax(self.softmax_out,axis=1),tf.int32)

            # 预测值和真实值(目标)对比
            correct_prediction = tf.equal(self.predict,tf.reshape(self.input_obj.targets,[-1]))
            # 计算预测的精度
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
            # 如果测试，直接退出
            if not is_training:
                return
            # 学习率
            self.learning_rate = tf.Variable(0.0,trainable=False)
            # 返回所有可被训练(trainable=True) 
            # 也就是除了不可训练的学习率之外其他变量
            tvars = tf.trainable_variables()

            # tf.clip_by_global_norm (实现 Gradient Clipping 梯度裁剪) 这是为了防止梯度爆炸
            # tf.gradiens 计算 self.cost 对于 tvars 的梯度（求导）返回一个梯度列表
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost,tvars),5)

            # 优化器
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

            # apply_gradients(应用梯度) 将之前用(Gradient Clipping) 梯度裁剪过的梯度应用到可被训练的变量上去，做梯度下降
            # apply_gradients 其实是 minimize 方法里面的第二步，第一步是计算梯度
            self.train_op = optimizer.apply_gradients(zip(grads,tvars),global_step=tf.train.get_global_step())
            # 用于更新学习率
            self.new_lr = tf.placeholder(tf.float32,shape=[])
            self.lr_update = tf.assign(self.learning_rate,self.new_lr)
    def assign_lr(self,session,lr_value):
        session.run(self.lr_update,feed_dict={self.new_lr:lr_value})

            