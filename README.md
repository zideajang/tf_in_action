
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
## 多层感知机(MLP)
层与层连接使用权重和偏置来进行连接
## 卷积神经网络
### 应用
- 图像识别
- 迁移学习
## 经典神经网络
### 
## 循环神经网络(RNN)
### 应用
- 机器翻译
- 问答系统
- 智能音箱（聊天机器）
- 通过学习谱曲和作词

### 为什么需要神经网络
- 是为了解决时间相关的序列数据问题
- 有些问题是现有的卷积神经网无法解决，因为卷积神经网用于解决空间问题而非时间上序列的问题。
- 对于长短不一数据也是 MLP 无法处理的。也可以通过补位(padding)来对数据进行处理
- 共享参数
![共享参数](https://upload-images.jianshu.io/upload_images/8207483-2d378b88cd36faae.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![RNN展开](https://upload-images.jianshu.io/upload_images/8207483-0ac1aa5e5871c5be.jpeg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

为了增加神经网深度，并且降低参数。

### 参数共享、动态系统
### 带输入的动态系统
$$ s^{(t)} = f(s^{(t-1)};\theta)$$
$$ h^{(t)} = f(h^{(t-1)},x^{(t)};\theta)$$
### 标准循环神经网络
- 输入 x 和输出 y 都是向量的序列

$$ a^{(t)} = b + Wh^{(t-1)} + Ux^{(t)} $$

隐藏层的类型
- 原始的 RNN 结构会出现梯度消失和梯度爆炸的问题
- 使用LSTM和GRU可以避免这些问题

### 序列映射到向量
- 情感分析(Sentiment analysis)
### 增加输出与隐藏状态的联系
- 词性标志(Part of Speech tagging 或 POS tagging)
### 向量映射到为序列
- 图像语义分析(image captioning)

## 语言模型
### 语言模型(Word Embedding)
- Non-NN 模型
- NNLM
- Word2Vec
- ELMo
- BERT
- TF-hub
- Bert As Service

### 为什么需要
- 输入特征需要进行数值转换，unicode 编码就是词的 id 并不包含词语义上相关性，如果我们需要考虑词与词的相关性时候。我们掌握了一个词汇其实就是掌握了符号与语义的映射。这是我们日积月累得出
### One Hot Encodiing
- 本质就是一个向量
- 是一个全体词量大小维度的向量，其中只有一个位置为 1 其他都是 0 来表示一个单词
- 问题是维度比较大，并且没有语义。
- 而且要是做向量乘法，计算量也是非常大
- 每一个向量都是独立正交的。

### Bag-of-Words(Bow) 模型
- 词袋模型是把一个句子中出现的单词次数进行统计
- <word,count> map
- 就可以比较两个向量，通过 cos 也就是两个向量夹角来比较两个向量的相似度
- TF-IDF 如果在文章比较稀有说明单词比较重要。

### 基于神经网(NNLM)
- one-hot 编码和 BoW 编码其实都是相互独立，都是向量正交
- 其实我们把一些单词尽心归类，例如单词不同时态
    - 名称
    - 形容词

- 示例,这个比较经典
[girl,woman,boy,man]

|  | 0 | 1 |
| ------ | ------ | ------ |
| gender | female | male |
| age | child | adult |

- 其实 $f$ 就是说我们的 word embedding，手动标记，自然语言难就在于其多义性
- 我们需要让计算机来学习这些不同维度，
- 需要使用非监督学习，可以使用 GAN 来实现 autoencoder 来获取  z 也就是获取一定维度向量来表示词

### NNLM(Nenural Network Base)
- 通过上下文，根据上文推测下文，作为副产物用于训练 word embedding

### Word2Vec
- 2013 有 google 的大神 Tomas Mikolov Jeffrey Dean
- 类似于 NNLM 但是关注 word embedding
- 两种方式
    - 持续 Bag-of-Word(CBOW)
    - 持续 Skip-gram (Skip-gram)
#### CBOW
- 单词是在上下文中才有意义

### GloVe
- Global Vectors for Word Representation
- 上下文并不是限于一定范围，而是放大全文
- 无法实现在线训练

### ELMo
- 在 2018 年出现了 ELMo 解决自然语言痛点的多义词。
- 在 ELMo 之前 word embedding 都是静态
- 在 ELMo 提供我们一些变量，可以得到动态的 embedding
- 至少需要 2 层 LSTM 来得到 3 个 embedding
- 从上文预测目标词，从下文来预测目标词，是双向的来训练模型
- 通过调整三个 embedding 的权重来动态改变单词的向量
- 所以现在的 embedding 是预训练需要我们自己网络使用 fine tuning

### BERT
- Transform 采取特征提取器
- Attention Is All You Need
- OpenAI GPT

### 

## Transform
- Seq2Seq model 特别 Self-attention
### Self-attention 
- q: $ q^i = W_i a^i $

- 拿每一个 query q 去对每一个 k 做 attention
- $$ a_{1,i} =  \frac{q^1 \cdot k^i}{ \sqrt{d}}  $$
  q 和 k 多点乘后variance 越大，
- 做 softmax 
    - $$\hat{\alpha} = $$
- $$ b^1 = \sum_i \hat{\alpha}_{1,i} v^i$$

### Sequence
- RNN 不容易被并行化，这个应该很好理解
- 用 CNN 代替 RNN



#### 计算成本高
- Huffman 编码和 Softmax 归一化运算量比较大
- 负采样
    we play game(neg:eat)
- softmax 和 sigmoid 神经网引入激励函数

#### 向量空间
France - Paris + Beijing = China 
可以做一些文字间的加减法


#### Skip-gram




### 循环神经网络
### 多层网络与双向网络
### 长短期记忆网络(LSTM)
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