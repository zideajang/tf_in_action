
### 引言
- 框架的重要性
我们在学习机器学习和实践过程中，做一些基础工作有助了解其原理和一些模型设计机制。但是使用框架好处是不然而喻的，通过使用框架提供方法可以省去我们自己造轮子的时间，让我们更快开发模型，更加专注网络模型的设计。
- 为什么是 TensorFlow
现在网上大部分机器学习示例中使用框架多半都是 TensorFlower，当然现在除了 TensorFlow 以外您还有许多选择。首先区别于其他框架 TensorFlow 有 TensorBoard 这个好用的可视化工具，其实机器学习神经网络结构是如何通过网络结构来训练模型，这一切对于我们来说是有点神秘的，有时候我们想要了解机器是如何通过一层一层神经网络来学习的。这时候 TensorBoard 就派上用场了。
### 
- 图运算: 能够更好地表达复杂运算，让错综复杂关系一目了然
- 
### 基本要求

## 基础
## 线性回归
## 多层感知机
## 卷积神经网络
## 经典神经网络
### 
## 循环神经网络
## STML
## 对抗神经网络

### 课程设计
我会不断更新TensorFlow 基础分享，因为自己也在不断学习更新自己对机器学习认识，同时每天也在刷新自己对 TensorFlow 框架认识，希望大家和我不断地探索深度神经网络。

### 如何学习
官方文档是最好文档，方便你查询 Tensorflow 的 API 使用方法。我们除了可以在官方文档使用，我们可以在终端里来通过 help(查看的对象) 来获取对应 api 的帮助文档。具体操作如下
```
>>> import tensorflow as tf
>>> help(tf.Tensor)
```

> 参考书籍
《TensorFlow 深度学习》
《TensorFlow 学习指南》
《TensorFlow 实战 Google 深度学习框架》


### 安装 TensorFlow

学习新语言也好、新框架也好，我们都喜欢写一个 Hello World 开始。有关 Tensorflow 安装网上介绍版本很多，与其他 python 库安装不同的，也是值得注意的就是 Tensorflow 有 GPU 和 CPU 两个版本，推荐使用 GPU，不过在分享的实例中没有特别说明的我们都是使用 CPU。
#### 第一个 Hello TensorFlow
```
import tensorflow as tf
```
定义常量为 Hello TensorFlow。
```
hello = tf.constant("Hello TensorFlow")
sess = tf.Session()
print(sess.run(hello))
```
如果输出一下hello  ```<tf.Tensor: id=0, shape=(), dtype=string, numpy=b'hello'>``` 可以看出这是一个 Tensor 对象。
在 TensorFlow 中所有的计算都需要在Session（会话）中完成。这个大家一定要留心，否则就会出现问题得不到想要的效果。

#### 第二个 Hello TensorFlow
```
h = tf.constant("Hello")
t = tf.constant("TensorFlow")

ht = h + t

with tf.Session() as sess:
    result = sess.run(ht)

print(result)
```


```
h = tf.constant("Hello")
w = tf.constant(" world")

hw = tf.add(h,w)

mul = tf.multiply(a,b)
with tf.Session() as sess:
    result = sess.run(hw)
print(result)

```

```
mul = tf.multiply(a,b)
with tf.Session() as sess:
    result = sess.run(mul)
print(result)
```
#### TensorFlow 的操作符
定义操作符
```
a = tf.constant(5)
b = tf.constant(3)
c = tf.constant(2)

print(a)
```
```
Tensor("Const:0", shape=(), dtype=int32)
```
输出一个 a 的类型为 Tensor ，Tensor 类似一个对象，dtype 为数据类型。

```
mul = tf.multiply(a,b)
```
在 TensorFlow 中我们运算操作定义有区别于其他我们熟悉语言，输出 mul 为一个对象，而非 a 和 b 乘积的结果，而是 Tensor 节点，需要执行运算后才能得到结果。如果写成 a * b 则是  tf.multiply(a,b) 缩写。
```
Tensor("Mul:0", shape=(), dtype=int32)
```
使用 with 语句打开会话能够保证一旦所有计算完成后会话将自动关闭。

```
a = tf.constant(5)
b = tf.constant(3)
c = tf.constant(2)

# f =  (a + b) * c

ab = tf.add(a,b)
f = tf.multiply(ab,c)

with tf.Session() as sess:
    result = sess.run(f)
print(result)

```
当我们输入 Tensor（在这里可以叫张量或者节点）到会话进行计算，方法调用时会完成一组运算，从该节点根据依赖进行计算出所有节点。