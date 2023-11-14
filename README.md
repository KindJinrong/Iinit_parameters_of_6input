# Iinit_parameters_of_6input
对于机器人领域中少输入，多神经元设置特定的权重初始化
[toc]

## 确定网络中初始化

基于希望方差能无衰减的传递准则，Math：

$$
Var(z)=\sum_{1}^{n} Var(x_i) Var(w_{i})={\color{red}nVar(w)}Var(x)
$$
我们对于权重矩阵 我们设定特地的初始化分布，以均匀分布举例，易得正态分布

Solution1:

均匀分布方差 $$Var(x)=\frac{(a-b)^2}{12}$$,$U\sim[a,b]$ 且$a=-b$

$${\color{red}nVar(w)=1}，Var(w)=1/n=\frac{(a-b)^2}{12}$$

注意 一般会有一个增益系数（超参数），使用pytorch的推荐置，gain，使用tanh 则为$\frac{5}{3}$

kaiming 何凯明

$x\sim U[-\sqrt\frac{3}{\text{\color{red}fan-in}} , \sqrt\frac{3}{\text{\color{red}fan-in}} ]$

Xavier 泽维尔

$x\sim U[-\sqrt\frac{6}{\text{\color{red}fan-in+fan-out}} , \sqrt\frac{6}{\text{\color{red}fan-in+fan-out}} ]$

对于属性比较少的数据，显然不能使用 Xavier初始化，因为会导致数值方差过小，数据过小，表示能力不足。（准则是在神经网络中，我们认为数值大一点，方差大一点，能包含更多信息），且偏置 初始化为 均匀分布，使得模型更有表现力。



---



## 激活函数选择具有更大表现力的 sigmoid 或者 tanh，

选择==tanh==的原因

1.   数值初始化 是$N(0|1)$，有负数
2.   $tanh(x)=2sigmoid(2x)-1$，俩者特点近似，
3.   连续光滑、严格单调
4.   输出范围为(-1,1)，以(0, 0)为对称中心，均值为0；
5.    输入在0附近时，输出变化明显；输入离0越远，输出变化越小最后输出趋近于1不变。
6.   可以缓解zigzag现象
7.   梯度取值范围比sigmoid导数更广一些，可以缓解梯度消失（本身网络也短小精悍）。

---

## 惩罚项

仅对权重参数进行惩罚，而不对偏置进行惩罚。因为偏置控制的是函数的位置，而不是拟合程度。如果惩罚会容易导致欠拟合。
