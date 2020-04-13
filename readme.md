这个文件夹的代码是有关RRCF改进实验的代码
实验总共分为一下几步：


1. contest_data文件夹中存放所有的KPI，每条KPI用一个CSV文件存储
   CSV文件有3列，第一列是timestamp，第二列叫value是KPI数据，第三列叫anomaly是异常标签
2. rrcf_test文件夹存放了所有的代码和实验结果
   2.1 active文件夹存放的是主动推荐的点。 推荐的点按照实验编号划分，例如实验5.1的点在文件夹5.1当中
   2.2 contest_cluster_pic文件夹存的是聚类之后每一类KPI画的图像
   2.3 contest_data文件夹里面存储的是实验结果，实验结果按照实验编号划分，例如实验5.1的结果在文件夹中5.1中
   2.4 contest_pic文件夹里面存储的是可视化的实验结果
   2.5 best_F1.py用来计算best F1-score
   2.6 Clustering.py对KPI进行聚类，KPI和KPI的相似度使用TSsimilarity文件夹中的代码计算
   2.7 Donut.py 是异常检测算法Donut的一个实现
   2.8 donut.sh Donut.py只能检测一条kpi，这个shell脚本多次调用Donut.py，检测多条KPI
   2.9 evaluation.py 计算F1-score, precision, recall，被best_F1.py多次调用计算best F1-score
   2.10 Isolation_forest.py实现了Isolation Forest算法对KPI进行异常检测
   2.11 plot.py 画出实验结果的可视化图，存储在contest_pic文件夹当中
   2.12 Random Forest.py 实现了Random Forest算法对KPI进行异常检测
   2.13 rrcf.py 实现了RRCF算法，并且实现了对RRCF算法的改进
   2.14 run.sh 把所有的KPI分成多个batch, 多次调用test.py对不同的KPI batch进行测试
   2.15 setting.py 存储了很多布尔变量， 每个布尔变量控制一个优化是否开启
   2.16 test.py 将KPI分成很多batch。test.py调用了rrcf.py中的RRCF算法每次对一个batch进行测试
   2.17 utils.py 存了一些功能，被test.py调用。这些功能包括对KPI进行归一化、 切分KPI的训练集和测试集、
		提取特征、计算对象的大小、补全KPI的空缺点
3.TSsimilarity 计算KPI之间的相似性，这部分代码是借用的别人的，具体代码使用方式看这个文件夹里的readme