# Deep learning terminology explanation

### SOTA(State of the arts)

  在某一个领域做的Performance最好的model，一般就是指在一些[benchmark](https://so.csdn.net/so/search?q=benchmark&spm=1001.2101.3001.7020)的数据集上跑分非常高的那些模型。

- **SOTA mode**l并不是特指某个具体的模型，而是指在该项研究任务中，目前最好/最先进的模型。
- **SOTA result**指的是在该项研究任务中，目前最好的模型的结果/性能/表现。

****

### 非端到端模型

  传统机器学习的流程往往由多个独立的模块组成，比如在一个典型的自然语言处理（Natural Language Processing）问题中，包括分词、词性标注、句法分析、语义分析等多个独立步骤，每个步骤是一个独立的任务，其结果的好坏会影响到下一步骤，从而影响整个训练的结果，这是非端到端的。

****

### 端到端模型

  从**输入端到输出端**会得到一个预测结果，将预测结果和真实结果进行比较得到误差，将误差反向传播到网络的各个层之中，调整网络的权重和参数直到模型收敛或者达到预期的效果为止，中间所有的操作都包含在神经网络内部，不再分成多个模块处理。由原始数据输入，到结果输出，从输入端到输出端，中间的神经网络自成一体（也可以当做黑盒子看待），这是端到端的。


### Benchmark、Baseline

**Benchmark**和**baseline**都是指最基础的比较对象。

论文的motivation来自于想超越现有的baseline/benchmark，你的实验数据都需要以**baseline/benckmark**为基准来判断是否有提高。

唯一的区别：

- baseline讲究一套方法

- benchmark更偏向于一个目前最高的指标，比如precision，recall等等可量化的指标。

  举个例子，NLP任务中BERT是目前的SOTA，你有idea可以超过BERT。那在论文中的实验部分你的方法需要比较的baseline就是BERT，而需要比较的benchmark就是BERT具体的各项**指标**。

****

### 迁移学习

  迁移学习通俗来讲，就是运用**已有的知识**来学习**新的知识**，核心是找到已有知识和新知识之间的**相似性**。
****

### 微调

微调其实讲的是利用原有**模型参数**（“知识”）初始化现有模型，在此基础上继续train自己的model（“再加工”）。```说人话就是把现成的模型略加修改然后再作少量training，主要用于样本数量不足的情形。```

****

### 监督学习

  是使用足够多的**带有label**的数据集来训练模型，数据集中的每个样本都带有人工标注的label。通俗理解就是，模型在学习的过程中，“老师”指导模型应该向哪个方向学习或调整。

****

### 非监督学习

  是指训练模型用的数据**没有人工标注的标签信息**，通俗理解就是在“没有老师指导”的情况下，靠“学生”自己通过不断地探索，对知识进行归纳和总结，尝试发现数据中的内在规律或特征，来对训练数据打标签。

****

### 半监督学习

  是在只能获取**少量的带label**的数据，但是**可以获取大量的的数据**的情况下训练模型，让学习器不依赖于外界交互，自动地利用未标记样本来提升学习性能，半监督学习是监督学习和非监督学习的相结合的一种学习方法。

****

### 泛化（Generalization）

​	模型的泛化能力通俗易懂的说就是模型在测试集（其中的数据模型以前没有见过）中的表现，也就是模型举一反三的能力，但是这些数据必须满足与iid（独立同分布）并在同一个分布中。
举个例子：一张图片模型之前没有见过，但是这张图片与TrainDataSet在同一分布，并满足iid，模型可以很好的预测这张图，这就是模型的泛化，在测试集中，模型预测新数据的准确率越高，就可以说是模型的泛化能力越好。

****

### 大模型

一般指1亿以上参数的模型，但是这个标准一直在升级，目前万亿参数以上的模型也有了。大语言模型（Large Language Model，LLM）是针对语言的大模型。

****

### 指令微调（Instruction FineTuning）

  针对已经存在的预训练模型，给出额外的指令或者标注数据集来提升模型的性能，如P-tuning， prompt-tuning，prefix-tuning。

****

### 增量微调

是指在神经网络中增加额外的层，如lora，adapter。

****

### 175B、60B、540B

这些一般指参数的个数，B是Billion/十亿的意思，175B是1750亿参数，这是GPT3的参数规模。

****

### 强化学习

(Reinforcement Learning）一种机器学习的方法，通过从外部获得激励来校正学习方向从而获得一种自适应的学习能力。

****

### 基于人工反馈的强化学习（RLHF)

（Reinforcement Learning from Human Feedback）构建人类反馈数据集，训练一个激励模型，模仿人类偏好对结果打分，这是GPT-3后时代大语言模型越来越像人类对话核心技术。

****

### 涌现

研究发现，模型规模达到一定阈值以上后，会在多步算术、大学考试、单词释义等场景的准确性显著提升，称为涌现。

****

### 思维链

（Chain-of-Thought，CoT）。通过让大语言模型（LLM）将一个问题拆解为多个步骤，一步一步分析，逐步得出正确答案。需指出，针对复杂问题，LLM直接给出错误答案的概率比较高。思维链可以看成是一种指令微调。

****

### Pooling Layer 池化层

​	池化层是神经网络中用于降低数据维度的层，它通过对输入数据进行下采样来减少计算量和过拟合。PyTorch中常用的池化层包括最大池化（Max Pooling）和平均池化（Average Pooling）。

- **Max Pooling 最大池化**

  最大池化层选择每个池化窗口中的**最大值**作为输出。它有助于提取输入数据中的关键特征，并在一定程度上实现平移不变性。在PyTorch中，可以使用`nn.MaxPool2d`类创建最大池化层。

- **Average Pooling 平均池化**

  平均池化层计算每个池化窗口中的平均值作为输出。与最大池化相比，平均池化更注重保留输入数据的整体信息。在PyTorch中，可以使用`nn.AvgPool2d`类创建平均池化层。

****

### Linear Layer 线性层

 线性层（也称为全连接层）是神经网络中用于**实现线性变换**的层。

```它将输入数据与权重矩阵相乘，并加上偏置项，以生成输出。```在PyTorch中，可以使用`nn.Linear`类创建线性层。

****

### Activation Function Layer 激活函数层

激活函数层用于引入非线性因素，使神经网络能够逼近复杂的函数。常见的激活函数包括ReLU、Sigmoid和Tanh等。在PyTorch中，可以使用`nn`模块中的相应类创建激活函数层。