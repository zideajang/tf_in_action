#encoding=utf-8

"""
"""
import tensorflow as tf

# 定义超参数
EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 0.0002
BETA_1 = 0.5


# 定义判别器模型

def discriminator_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(
        64, # 64 个过滤器，输出的深度是(depth) 是 64
        [5,5], # 过滤器在二维的大小是(5 * 5)
        padding='same', # same 表示输出大小不变，因此需要在外围补零 2 圈
        input_shape=(64,64,3) #输入形状 (64,64,3) 3 是表示 RGB
    ))

    model.add(tf.keras.layers.Activation("tanh")) # 添加 Tanh 激活层
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2))) #池化层

    model.add(tf.keras.layers.Conv2D(128,(5,5)))
    model.add(tf.keras.layers.Activation("tanh")) 
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2))) #池化层

    model.add(tf.keras.layers.Conv2D(128,(5,5)))
    model.add(tf.keras.layers.Activation("tanh")) 
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2))) #池化层

    model.add(tf.keras.layers.Flatten()) #扁平操作
    model.add(tf.keras.layers.Dense(1024)) # 1024 个神经元的全连接层
    model.add(tf.keras.layers.Activation("tanh")) 
    model.add(tf.keras.layers.Dense(1)) # 1 个神经元的全连接层

    model.add(tf.keras.layers.Activation("sigmoid")) # 添加 Sigmoid 层

    return model

# 定义生成器模型
# 从随机数生成图片
def generator_model():
    model = tf.keras.models.Sequential()
    # 输出维度是 100，输出维度(神经元个数) 是 1024 的全连接层
    model.add(tf.keras.layers.Dense(input_dim=100,units=1024)) # 接收随机数，输入维度 100
    model.add(tf.keras.layers.Activation("tanh"))
    model.add(tf.keras.layers.Dense(128 * 8 * 8)) # 8192 个神经元的全连接层
    # 数据标准化操作
    """
    平均的激活函数平均值趋近于 0
    """
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation("tanh"))
    # 8 x 8 像素图片
    model.add(tf.keras.layers.Reshape((8,8,128),input_shape=(128 * 8 * 8,)))
    # 16 x 16 像素图片
    model.add(tf.keras.layers.UpSampling2D(size=(2,2)))
    # 
    model.add(tf.keras.layers.Conv2D(128,(5,5),padding='same'))
    model.add(tf.keras.layers.Activation("tanh"))

    # 32 x 32 像素图片
    model.add(tf.keras.layers.UpSampling2D(size=(2,2)))
    model.add(tf.keras.layers.Conv2D(128,(5,5),padding='same'))
    model.add(tf.keras.layers.Activation("tanh"))

    # 64 x 64 像素图片
    model.add(tf.keras.layers.UpSampling2D(size=(2,2)))
    model.add(tf.keras.layers.Conv2D(3,(5,5),padding='same'))
    model.add(tf.keras.layers.Activation("tanh"))

    return model


# 构造一个 Sequential 对象，包含一个生成器和一个判别器
# 输入 -> 生成器 -> 判别器 -> 输出
def generator_containing_discriminator(generator,discriminator):
    model = tf.keras.models.Sequential()
    model.add(generator)
    discriminator.trainable = False #初始时判别器是不可被训练
    model.add(discriminator)

    return model




