from rnn_utils import *
from network import *

# 训练数据
"""
model_save_name 将模型参数保存到文件
train_data
vocab_size 10000
num_layers LSTM 层的数目
num_epochs 
batch_size 批次
model_save_name 每一个 epoch 都保存
learning_rate
max_lr_epoch 超过 10 后学习率会开始衰减
lr_decay
int_iter
"""
def train(train_data,vocab_size,num_layers,num_epochs,batch_size,model_save_name,learning_rate=0.1,max_lr_epoch=10,lr_decay=0.93,print_iter=50):
    # 训练输入
    training_input = Input(batch_size=batch_size,num_steps=35,data=train_data)
    # 创建训练的模型
    m = Model(training_input, is_training=True, hidden_size=650,vocab_size=vocab_size,num_layers=num_layers)

    # 初始化变量的操作
    init_op = tf.global_variables_initializer()

    # 初始化学习率
    orig_decay = lr_decay
    with tf.Session() as sess:
        sess.run(init_op) #初始化所有变量
        # coordinator(协调器)，用于协调线程的运行
        coord = tf.train.Coordinator()
        # 启动线程
        threads = tf.train.start_queue_runners(coord=coord)
        # 为了用 Saver 来保存模型的变量
        saver = tf.train.Saver() # max_to_keep 默认是保存最近 5 个模型参数文件
        # 开始 Epoch 的训练
        for epoch in range(num_epochs):
            # 只有 Epoch 数大于 max_lr_epoch(设置为 10)后，才会使学习率衰减
            new_lr_decay = orig_decay ** max(epoch + 1 - max_lr_epoch,0.0)
            m.assign_lr(sess,learning_rate * new_lr_decay)
            # 当前的状态
            # 第二维是 2 是因为对每一个 LSTM 单元有两个来自上一个单元的输入
            # 一个是前一时刻 LSTM 的输出 $h_{t-1}$
            # 一个是前一时刻的单元状态$C_{t-1}$

            current_state = np.zeros((num_layers,2,batch_size,m.hidden_size))

            # 获取当前时间，以便打印日志时用
            curr_time = datetime.datetime.now()

            for step in range(training_input.epoch_size):
                if step % print_iter != 0:
                    cost,_,current_state = sess.run([m.cost,m.train_op,m.state],feed_dict={m.init_state:current_state})
                else:
                    seconds = (float(datetime.datetime.now() - curr_time).seconds) / print_iter
                    curr_time = datetime.datetime.now()
                    cost, _, current_state, acc = sess.run([m.cost,m.train_op,m.accuracy],feed_dict={m.init_state:current_state})
                    print("Epoch {} , step {} Cost: {:.3f}, Accuracy:{:.3f}, Seconds per step:{:.3f}".format(epoch,step,cost,acc,seconds))
            saver.save(sess,save_path +'/' + model_save_name, global_step=epoch)
        saver.save(sess,save_path +'/' + model_save_name + '-final')
        coord.request_stop()
        coord.join(threads)
if __name__ == "__main__":
    if args.data_path:
        data_path = args.data_path
    train_data,valid_data,test_data,vocab_size,id_to_word = load_data()
    train(train_data,vocab_size,num_layers=2,num_epochs=70,batch_size=20,model_save_name='train-checkpoint',)