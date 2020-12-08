# CTR_Model
-i https://pypi.tuna.tsinghua.edu.cn/simple


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# jupyter notebook

### 在CTR预估以及推荐系统等场合下：

> LR: LR最大的缺点就是无法组合特征，依赖于人工的特征组合，这也直接使得它表达能力受限，基本上只能处理线性可分或近似线性可分的问题。

> FM: FM通过隐向量latent vector做内积来表示组合特征，从理论上解决了低阶和高阶组合特征提取的问题。但是实际应用中受限于计算复杂度，一般也就只考虑到2阶交叉特征。后面又进行了改进，提出了FFM，增加了Field的概念。

> CNN: CNN模型的缺点是：偏向于学习相邻特征的组合特征。

> RNN: RNN模型的缺点是：比较适用于有序列(时序)关系的数据。

> FNN: 先使用预先训练好的FM，得到隐向量，然后作为DNN的输入来训练模型。缺点在于：受限于FM预训练的效果，Embedding的参数受FM的影响，不一定准确；预训练阶段增加了计算复杂度，训练效率低; FNN只能学习到高阶的组合特征。模型中没有对低阶特征建模。

> PNN: PNN为了捕获高阶组合特征，在embedding layer和first hidden layer之间增加了一个product layer。但是内积的计算复杂度依旧非常高，原因是：product layer的输出是要和第一个隐藏层进行全连接的;product layer的输出需要与第一个隐藏层全连接，导致计算复杂度居高不下;和FNN一样，只能学习到高阶的特征组合。没有对于1阶和2阶特征进行建模。

> Wide&Deep：同时学习低阶和高阶组合特征，它混合了一个线性模型（Wide part）和Deep模型(Deep part)。这两部分模型需要不同的输入，而Wide part部分的输入，依旧依赖人工特征工程。
但是，这些模型普遍都存在一个问题：偏向于提取低阶或者高阶的组合特征。不能同时提取这两种类型的特征。 需要专业的领域知识来做特征工程。无论是FNN还是PNN，他们都有一个绕不过去的缺点：对于低阶的组合特征，学习到的比较少。

> DeepFM：在Wide&Deep的基础上进行改进，不需要预训练FM得到隐向量，不需要人工特征工程，能同时学习低阶和高阶的组合特征；FM模块和Deep模块共享Feature Embedding部分，可以更快的训练，以及更精确的训练学习。
