# Transformer

一种用于（NLP）自然语言处理和其他序列到序列（Seq 2 Seq）任务的深度学习模型框架

****

### Consist Part

-  **自注意力机制（Self-Attention）**：可以使模型能够同时考虑序列中的所有位置，而不是想循环神经网络（RNN）或者卷积神经网络（CNN）一样逐步处理。自注意力机制允许模型根据输入序列中不同部分来赋予不同的注意权重，从而更好的捕捉语义关系
- **多头注意力（Multi-Head Attention）**：Transformer中的自注意力机制被扩展为多个注意力头，每个头可以学习不同的注意权重，以更好地捕捉不同类型的关系。多头注意力允许模型并行处理**不同的信息子空间**。
- **堆叠层（Stacked Layer）**：Transformer通常由多个相同的编码器和解码器堆叠而成。这些堆叠的层有助于模型学习复杂的特征表示和语义。

- **位置编码（Position Encoding）**:由于Transformer没有内置的序列位置信息，它需要额外的位置编码来表达输入序列中单词的位置顺序。
- **残差连接和层归一化（Residual Connections and Layer Normalization）**：这些技术有助于减轻训练过程中的梯度消失和爆炸问题，使模型更容易训练。
- **编码器和解码器：**Transformer通常包括一个编码器用于处理输入序列和一个解码器用于生成输出序列，这使其适用于序列到序列的任务，如机器翻译。

****

### Structure

总体结构：

![img](https://img-blog.csdnimg.cn/5d74c1e4fd7c435a914778542258b1de.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5ZGK55m95rCU55CDfg==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

![img](https://img-blog.csdnimg.cn/img_convert/816a3c48fa200e4da230192333ad76df.png)

**简略结构**：

![在这里插入图片描述](https://img-blog.csdnimg.cn/05cb46864afb474c951b0dc9a883b2d5.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5ZGK55m95rCU55CDfg==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

> Transformer的编码组件是由6个编码器叠加在一起组成的，解码器也是。
>
> 所有的编码器在结构上是相同的，但是它们之间**没有共享参数**

****

### 自注意力机制原理：

- **嵌入**

  模型将输入序列当中的每个单词嵌入到一个高维向量表示中，这个嵌入过程允许模型捕捉单词之间的语义相似性

- **查询，键和值向量**

  模型为每个单词计算三个向量;

  - 查询向量：表示单词的查询，模型在序列中寻找的内容
  - 键向量：表示单词的键，序列中其他单词应该注意的内容
  - 值向量：单词对于输出所贡献的信息

- **注意力分数**

  一旦模型计算了每个单词的查询、键和值向量，它就会为序列中的每一对单词计算注意力分数。这通常通过取查询向量和键向量的点积来实现，以评估单词之间的相似性。

- **SoftMax归一化**

  然后，使用 softmax 函数对注意力分数进行归一化，以获得注意力权重。这些权重表示每个单词应该关注序列中其他单词的程度。注意力权重较高的单词被认为对正在执行的任务更为关键。

- **加权求和**

  最后，使用注意力权重计算值向量的加权和。这产生了每个序列中单词的[自注意力机制](https://so.csdn.net/so/search?q=自注意力机制&spm=1001.2101.3001.7020)输出，捕获了来自其他单词的上下文信息。

![img](https://img-blog.csdnimg.cn/img_convert/ec2ab30f8b862d2b7197bd8534cc86f0.png)