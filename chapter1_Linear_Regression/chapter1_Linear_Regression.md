
## Andrew Ng机器学习（一）：线性回归
机器学习公开课笔记以及作业题分析

***
## 主要内容
### 初步介绍
__监督式学习__: 给定数据集并且知道其正确的输出应该是怎么样的，即有反馈（feedback），分为
* 回归 （Regressioin）: map输入到连续的输出值
* 分类 （Classification）：map输出到离散的输出值

__非监督式学习__: 给定数据集，并不知道其正确的输出是什么，没有反馈，分为
* 聚类（Clustering）： Examples: Google News, Computer Clustering, Markert Segmentation
* 关联（Associative）：Examples: 根据病人特征估算其病症

***

### 一元线性回归
* 假设（Hypothesis）：$h_{\theta}(x)=\theta_0+\theta_1x$
* 参数（Parameters）：$\theta_0, \theta_1$
* 代价函数（Cost Function）：$J(\theta_0, \theta_1)=\frac1{2m}\displaystyle{\sum_{i=1}^{m}} (h_\theta(x^{(i)})-y^{(i)})^2$（最小二乘法）
* 目标函数（Goal）: $\displaystyle{\min_{\theta_0, \theta_1}}J(\theta_0, \theta_1)$

***

### 梯度下降算法（Gradient descent）
__基本思想__：
* 初始化$\theta_0, \theta_1$
* 调整$\theta_0, \theta_1$直到$J(\theta_0, \theta_1)$达到最小值，更新公式$\theta_j=\theta_j-\alpha\frac{\partial}{\partial\theta_j}J(\theta_0,\theta_1)$
对于一元线性回归问题，对$J(\theta_0, \theta_1)$求偏导数可得
$$\frac{\partial{J}}{\partial\theta_0}=\frac1{2m}\displaystyle{\sum_{i-1}^{m}}2\times(\theta_0+\theta_1x^{(i)}-y^{(i)})=\frac1{2m}\displaystyle{\sum_{i=1}^m}(h_\theta(x^{(i)})-y^{(i)})$$
$$\frac{\partial{J}}{\partial\theta_1}=\frac1{2m}\displaystyle{\sum_{i-1}^{m}}2\times(\theta_0+\theta_1x^{(i)}-y^{(i)})x^{(i)}=\frac1{2m}\displaystyle{\sum_{i=1}^m}(h_\theta(x^{(i)})-y^{(i)})x^{(i)}$$
从而参数$\theta_0, \theta_1$的更新公式为
$$\theta_0=\theta_0-\alpha\frac1m\displaystyle{\sum_{i=1}^m}(h_\theta(x^{(i)})-y^{(i)})$$
$$\theta_1=\theta_1-\alpha\frac1m\displaystyle{\sum_{i=1}^m}(h_\theta(x^{(i)})-y^{(i)})x^{(i)}$$
其中$\alpha$称为学习速率（learning rate），如果其太小，则算法收敛速度太慢；反之，如果太大，则算法可能会错过最小值，甚至不收敛。另一个需要注意的问题是，上面$\theta_0, \theta_1$的更新公式用到了数据集的全部数据 （称为“Batch” Gradient Descent），这意味着对于每一次update，我们必须扫描整个数据集，会导致更新速度过慢

***

### 线性代数复习
* 矩阵和向量定义
* 矩阵加法和数乘
* 矩阵-向量乘积
* 矩阵-矩阵乘积
* 矩阵乘法的性质：结合律，交换律不成立
* 矩阵的逆和转置：不存在逆元的矩阵称为“奇异（singular）矩阵”

***

### 多元线性回归
一元线性回归只有一个特征$x$，而多元线性回归可以有多个特征$x_1,x_2,...,x_n$
* 假设 (Hypothesis)：$h_\theta(x)=\theta^Tx=\theta_0x_0+\theta_1x_1+\theta_2x_2+...+\theta_nx_n$
* 参数 (Parameters)：$\theta_0,\theta_1,..,\theta_n$
* 代价函数 (Cost function)：$J(\theta_0,\theta_1,..,\theta_n)=\frac1{2m}\displaystyle{\sum_{i=1}^m}(h_\theta(x^{(i)})-y^{(i)})^2$
* 目标 (Goal)：$min_\theta J(\theta)$

***

### 梯度下降 (Gradient Descent)
迭代更新参数$\theta:\theta_j=\theta_j-\alpha\frac1m\displaystyle{\sum_{i=1}^m}(h_\theta(x^{(i)})-y^{(i)})x^{(i)}~~,j=0,1,2,...,n $

向量化实现 (Vectorized Implementation)：$\theta=\theta-\alpha\frac1m(X^T(X\theta-y))$

__特征缩放（Feature Scaling）__

动机：如果不同特征之间的数值量级差别太大，那么梯度下降的速度非常慢，为了加快算法的收敛速度，将各个特征划归到统一数量级，一般是[0, 1]或者[-1, 1]之间

Trick1:$x_j=\frac{x_j-\mu_j}{s_j}$, 其中$\mu_j$表示第j个特征的均值，$s_j$表示第j个特征的范围(max
-min)或者标准差(standard deviation)

Trick2:学习速率的选择
* 合理的选择学习速率，保证$J(\theta)$的值在每一次迭代后都是下降的
* 如果$J(\theta)$随迭代次数单调递增或者$J(\theta)$随迭代次数成波浪形(例如: \/\/\/\/\/\/), 这时候应该考虑选择较小的$\alpha$; 但是$\alpha$太小会导致收敛速度过慢
* 为了正确的选择$\alpha$，尝试序列 0.001, 0.01, 0.1, 1等

***

### 多项式回归（Polynomial Rregression）
如果用线性不能很好地拟合数据，可以考虑多项式。例如
$$h(x)=\theta_0+\theta_1x_1+\theta_2x^2+\theta_3x^3$$
有个技巧是令
$$x_2=x_2^2$$
$$x_3=x^3_3$$
这样便转化为线性回归的问题

***

### Normal Equation的数学推导
解析推导过程:
$$J(\theta)=\frac1{2m}\displaystyle{\sum_{i=1}^m}(h_\theta(x^{(i)})-y^{(i)})^2$$
可以简化写成向量的形式:
$$J(\theta)=\frac1{2m}||X\theta-y||^2=\frac1{2m}(X\theta-y)^T(X\theta-y)$$
展开可得:
$$J(\theta)=\frac1{2m}(\theta^TX^TX\theta-y^TX\theta-\theta^TX^Ty+y^Ty)$$
注意到$y^TX\theta$是一个标量，因此它与其转置$\theta^TX^Ty$是相等的，即中间两项是相等的，从而$J(\theta)$可以进一步化简为:
$$J(\theta)=\frac1{2m}[\theta^TX^TX\theta-2\theta^TX^Ty+y^Ty]$$
对向量的求导与单变量的求导法则有诸多不同，这里不加证明给出如下两个重要的向量求导结论:
$$d(X^TAX)/dX=(dX^T/dX)AX+(d(AX)^T/dX)X=AX+A^TX$$
$$d(X^TA)/dX=(dX^T/dX)A+(dA/dX)X^T=IA+0X^T=A$$
根据结论(1), 第一项的求导结果为$2X^TX\theta$；根据结论(2)，第二项的求导结果为$-2X^Ty$；第三项不含θθ，求导结果当然为0，整合三项我们可以得到$J(\theta)$的导数$\frac{dJ(\theta)}{d\theta}$：
$$\frac{dJ(\theta)}{d\theta}=\frac1{2m}(xX^TX\theta-2X^Ty)$$
令该导数等于0，我们很容易得到
$$\theta=(X^TX)^{-1}X^Ty$$

***
## 作业题分析
__单变量线性回归__

需要计算代价函数，给出梯度下降法的实现，然后默认给出了一个代价函数在$\theta_0,\theta_1$的平面上的分布情况的surf图和等高线图，以下是作业中画的图：
1. 数据的散点图
![此处输入图片的描述](http://imglf2.nosdn.127.net/img/WnpMNEZyYld6WUd6TFJlUm5tdXdoaS80akhjdkhSSTI2S3VDVjBvV0VHNUdiVHhLWE4wUk9RPT0.png?imageView&thumbnail=500x0&quality=96&stripmeta=0&type=jpg)
2. 线性回归的结果
![此处输入图片的描述](http://imglf.nosdn.127.net/img/WnpMNEZyYld6WUd6TFJlUm5tdXdocmpTNEhsaEFLYlRGZ1JBYXFQY2pEamVPVWZ4ZFQ3eFdnPT0.png?imageView&thumbnail=500x0&quality=96&stripmeta=0&type=jpg)
3. 代价函数的等高线图
![此处输入图片的描述](http://imglf0.nosdn.127.net/img/WnpMNEZyYld6WUd6TFJlUm5tdXdoajcrL08vTXZvNUJSSXN5TFgzK1JBcndQbEV4S0pEQzhBPT0.png?imageView&thumbnail=500x0&quality=96&stripmeta=0&type=jpg)
自己添加了一个用来记录代价函数在迭代过程中的历史记录的函数，并将代价函数的历史记录也画在了等高线图中，可以直观看到代价函数的变化趋势
4. surf图（代价函数的分布面）
![surf图](http://imglf2.nosdn.127.net/img/WnpMNEZyYld6WUd6TFJlUm5tdXdob3Z1L3Z1cytSdWp2NmU5d09lNjI5WlVUZnZ3MHFsemlnPT0.png?imageView&thumbnail=1680x0&quality=96&stripmeta=0&type=jpg)

__多变量线性回归__

需要写特征缩放的函数，计算代价函数的函数，以及梯度下降法的实现

1. 代价函数的值J变化情况
![代价函数的值J变化情况](http://imglf1.nosdn.127.net/img/WnpMNEZyYld6WUd6TFJlUm5tdXdoZ1NvRW1YeXcxS0pjc045UkRKQWoxWUVaRGVVbDJGUzFRPT0.png?imageView&thumbnail=500x0&quality=96&stripmeta=0&type=jpg)

***
### 参考
[1]. [Andrew Ng机器学习课程笔记2——线性回归](http://www.yalewoo.com/andrew_ng_machine_learning_notes_2_linear_regression.html)

[2]. [机器学习公开课笔记(1)：机器学习简介及一元线性回归](http://www.cnblogs.com/python27/p/MachineLearningWeek01.html)

[3]. [机器学习公开课笔记(2)：多元线性回归](http://www.cnblogs.com/python27/p/MachineLearningWeek02.html)

