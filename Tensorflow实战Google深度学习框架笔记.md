#### Tensorflow实战Google深度学习框架笔记
@(Tensorflow)


----------
##### Chap3 TensorFlow入门
前两章主要讲的是深度学习简介和TensorFlow的安装事宜，直接从第三章开始笔记。

TensorFlow中的三个模型——计算模型（图Graph），数据模型（张量Tensor），运行模型（会话Session）
先简单介绍一下三个模型是什么，再记一下它们之间是如何协作的。
**计算模型（图Graph）**
首先，计算图Graph定义了一个“作用域”，不同图之间的张量和运算不会共享，可以通过`tf.Graph`函数来生成新的计算图。
TensorFlow中每个计算都是计算图上的一个节点，节点之间的边描述计算之间的依赖关系。（`计算图上每个节点都是一个运算`）

TensorFlow会把程序中定义的计算自动转化为计算图上的节点这种形式，系统会维护一个默认的计算图(`tf.get_default_graph()`)<也可以自己生成新的计算图。
**数据模型（张量Tensor）**
在TensorFlow中，所有数据都是通过张量的形式来表示，可以将tensor理解为多维数组：
- 零阶张量——标量；
- 一阶张量——向量；
- n阶张量——n维数组；

其实张量只是对计算结果的`引用`，本身并没有保存任何值。
```python
	import tensorflow as tf
    a = tf.constant([1, 2], dtype=tf.int8, name='a')
	b = tf.constant([3, 4], dtype=tf.int8, name='b')
	result = a + b
	print(result)
	# 结果为：Tensor("add:0", shape=(2,), dtype=int8)
```

但可以通过在会话Session中`tf.Session().run(tensor_name)`来得到张量引用的计算结果。
**运行模型（会话Session）**
TensorFlow程序分为两阶段，第一阶段是定义计算图中所有的计算，第二阶段是通过会话Session执行计算。
一般使用会话需要显示的调用会话生成函数和会话关闭函数，因为（会话管理TensorFlow程序运行时的所有资源，计算完成之后需要关闭会话来释放资源，如果会话没有关闭会出现资源泄漏的问题）。

    sess=tf.Session()
	sess.run(tensor_name)
	sess.close()

可以使用Python的上下文管理工具，避免因为忘记关闭会话或者因为异常情况导致关闭会话函数没有执行的情况。

    with tf.Session() as sess:
	    sess.run()


----------
了解了TensorFlow这三个模型，就可以愉快的开始编写TensorFlow程序了。
接着上文的，简单实现个相加代码。

    import tensorflow as tf
    a=tf.constant([1.0,2.0],name='a')
    b=tf.constant([3.0,4.0],name='b')
    result=a+b
    with tf.Session() as sess:
	    sess.run(result)
	    	   

----------
**TensorFlow实现三层全连接神经网络**
![Alt text](./1520216888759.png)
三层神经网络，输入层是个1*2的矩阵$X$，输入层到隐藏层是个2*3的矩阵$W_1$，隐藏层到输出层是个3*1的矩阵$W_2$，输出层得到一个实值。
输入层一般使用常量`tf.constant`来表示，参数矩阵$W$使用变量来表示`tf.Variable`
> TensorFlow 中最基本的单位是常量（Constant）、变量（Variable）和占位符（Placeholder）。常量定义后值和维度不可变，变量定义后值可变而维度不可变。在神经网络中，变量一般可作为储存权重和其他信息的矩阵，而常量可作为储存超参数或其他结构信息的变量。占位符（Placeholder）顾名思义，先占个位置，在运行时再通过feed_dict{}给这个占位符“喂”值。

**TensorFlow中训练神经网络过程：**
- 定义网络结构和前向传播的输出结果；
- 定义损失函数以及选择反向传播算法；
- 生成会话Session并且在训练数据上反复运行反向传播算法。
- 


----------


##### Chap4 深层神经网络
