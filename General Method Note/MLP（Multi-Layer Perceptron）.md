# MLP（Multi-Layer Perceptron）

也称人工神经网络（ANN，Artificial Neural Network）

除了输入输出层，它中间可以有多个隐层，最简单的MLP只含一个隐层，即三层的结构，如下图：

![image-20240703162727413](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20240703162727413.png)

- 输入层：

  输入什么就是什么，比如输入是一个n维向量，就有n个神经元。

- 隐藏层：

  - 与输入层是全连接的，假设输入层用向量X表示，则隐藏层的输出就是

    f(W1X+b1)，W1是权重（也叫连接系数），b1是偏置，函数f 可以是常用的sigmoid函数或者tanh函数（激活函数）：

    ![image-20240703162935103](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20240703162935103.png)

- 输出层：

  其实隐藏层到输出层可以看成是一个多类别的逻辑回归，也即softmax回归，所以输出层的输出就是softmax(W2X1+b2)，X1表示隐藏层的输出f(W1X+b1)。