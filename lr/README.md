### Distributed Logistic Regression

#### Analysis

本项目对行分割与列分割下的两种分布式LR进行了实现，并在理论分析与实际数据上对两种实现方式进行了对比。因为在模型的训练过程中，节点的计算耗时是类似的。所以本项目主要从网络IO量以及内存使用量两个方面对比列分割与行分割。以下是对分析过程中使用的记号进行说明，$K$代表Worker节点的个数，$N$代表数据集中实例的个数，$W$代表参数的数量，$R$代表训练的轮数，$M$代表数据集中非空特征的总个数，由于实际生产环境中数据往往是非常稀疏的，那么现在假设数据集中每个特征是否为空是相互独立的，那么它非空的概率$P=\frac{M}{W\times N}$。

1. 内存使用量

   两种数据切分方式在内存使用量上的最大差别就在于参数副本以及实例预测值与真实值内存副本占用量的大小。

   按行分割需要在每个Worker节点上保存一份模型参数，如果不在每个Worker节点上保存参数就无法计算Worker节点上实例的预测值，所以这种方式保存参数副本使用的内存量为$O(KW)$。由于按行分割只需要在Worker节点上保存对应实例的预测值与标签的内存副本，所以此部分占用的内存量为$O( N)$。即总的内存占用量为$O(KW + N)$。

   按列分割则只要在每个Worker节点上保存对应特征的参数，所以保存参数使用的内存大小为$O(W)$。由于按列分割需要在每个Worker实例中保存所有实例的预测值，此部分内存占用量为$O(KN)$。所以总的内存占用量为$O(W + KN)$。

2. 网络IO

   按行分割的方式中网络IO主要集中在计算出梯度后发送到Master节点以及从Master节点拉取最新的参数，因为有$K$个Worker节点且需要进行$R$轮训练，所以总的网络IO量为$O(RKW)$。

   按列分割的方式因为每个Worker节点只保存对应特征的参数，所以在计算实例预测值时需要对每个Worker上实例的预测值在Master节点进行合并，然后再广播到所有Worker节点中。在合并所有Worker节点中实例的预测值时，每个Worker节点发送到Master节点的网络IO量为$N\times\frac{W}{K}$，因为有$K$个Worker节点，所以计算实例预测值时的网络IO量为$O(R\times N\times \frac{W}{K} \times K) = O(RNW)$。在同步实例的预测值时需要向$K$个Worker节点发送数据，所以同步实例预测值时的网络IO量为$O(RKN)$，所以按列划分数据集的总网络IO量为$O(RNW + RKN)$。

3. 小结

   通过上述的分析可以发现，按行对数据集进行分割时总的训练开销受参数规模的影响更大，而按列划分数据的方式受数据集规模的影响更大。所以在参数规模大而数据集规模小的时候选用按列划分数据更高效，而当样本数远远大于参数规模时选用按行对数据集进行划分更高效。

#### Data Partition

<img src="C:\Users\Administrator\Desktop\DistributedSystem\lr\system_design\data_split.svg" style="zoom:150%;" />

#### Data Structure

![](C:\Users\Administrator\Desktop\DistributedSystem\lr\system_design\data_organization.svg)
