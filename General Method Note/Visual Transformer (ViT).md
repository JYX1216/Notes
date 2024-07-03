# Visual Transformer (ViT)

### 工作流程：

- 将一张图片分成patches
- 将patches铺平
- 将铺平后的patches的线性映射到更低维的空间
- 添加位置embedding编码信息
- 将图像序列数据送入标准Transformer encoder中去
- 在较大的数据集上预训练
- 在下游数据集上微调用于图像分类

### 模型组成：

- **Linear Projection of Flattened Patches(Embedding层)**

  > 通过卷积方式将数据降维 3 * 3 -> 2 * 2

  对于图像数据而言，其数据格式为[H, W, C]是三维矩阵明显不是Transformer想要的。所以需要先通过一个Embedding层来对数据做个变换。如下图所示，首先将一张图片按给定大小分成一堆Patches。以ViT-B/16为例，将输入图片(224x224)按照16x16大小的Patch进行划分，划分后会得到196个Patches。接着通过线性映射将每个Patch映射到一维向量中，以ViT-B/16为例，每个Patche数据shape为[16, 16, 3]通过映射得到一个长度为768的向量（后面都直接称为token）。[16, 16, 3] -> [768]

  在代码实现中，直接通过一个卷积层来实现。 以ViT-B/16为例，直接使用一个卷积核大小为16x16，步距为16，卷积核个数为768的卷积来实现。通过卷积[224, 224, 3] -> [14, 14, 768]，然后把H以及W两个维度展平即可[14, 14, 768] -> [196, 768]，此时正好变成了一个二维矩阵，正是Transformer想要的。

  ![image-20240703160649930](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20240703160649930.png)

- Transformer Encoder

  Transformer Encoder其实就是重复堆叠**Encoder Block** L次，主要由Layer Norm、Multi-Head Attention、Dropout和MLP Block几部分组成。

  - **Linear**: 线性变换
  - **GELU**：激活函数（使得输出连续，保留线性特征与非线性特征）
  - **Dropout**: 正则化技术，防止模型过拟合
    - 每一次前向传播过程中，Dropout 会以一个预先设定的概率 ppp（称为 dropout rate）随机选择并忽略（即将其输出置为零）一部分神经元。被忽略的神经元在反向传播中也不更新权重。

  ![image-20240703161100838](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20240703161100838.png)

- MLP Head（最终用于分类的层结构）

  通过Transformer Encoder后输出的shape和输入的**shape是保持不变的**，以ViT-B/16为例，输入的是[197, 768]输出的还是[197, 768]。这里我们只是需要分类的信息，所以我们**只需要提取出[class]token**生成的对应结果就行，即[197, 768]中抽取出[class]token对应的[1, 768]。接着我们通过MLP Head得到我们最终的分类结果。MLP Head原论文中说在训练ImageNet21K时是由Linear+tanh激活函数+Linear组成。但是迁移到ImageNet1K上或者你自己的数据上时，只用一个Linear即可。


![image-20240703155913496](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20240703155913496.png)