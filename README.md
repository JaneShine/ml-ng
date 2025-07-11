# 机器学习课程笔记本集合

斯坦福吴恩达机器学习课程的个人学习notebook。

## 笔记本概览

### Week 1: 梯度下降 (Gradient Descent)
- **文件**: `week1-GradientDecent.ipynb`
- **内容**:
  - 线性回归模型公式推导与原理
  - 梯度下降算法原理与实现
  - 损失函数（cost function）定义与代码实现
  - 最小二乘法拟合与可视化
  - 参数空间的损失函数可视化
  - 学习率选择策略与收敛性分析
  - 数据生成与可视化

### Week 2: 学习率 (Learning Rate)
- **文件**: `week2-LearningRate.ipynb`
- **内容**:
  - 学习率（Learning Rate）对参数收敛的影响与手动调整方法
  - 梯度下降中对不同参数设置不同学习率的经验做法
  - 线性回归的正规方程（Normal Equation）推导与公式
  - 正规方程的适用范围与局限性说明

### Week 3: 逻辑回归与正则化 (Logistic Regression & Regularization)
- **文件**: `week3-LogisticRegression&Regularization.ipynb`
- **内容**:
  - 逻辑回归（Logistic Regression）原理与sigmoid函数推导
  - sigmoid函数的代码实现与可视化
  - 决策边界（Decision Boundary）理论与公式
  - 逻辑回归模型表达式与多特征输入
  - 理论数据生成与分类可视化
  - 阈值判定与分类规则
  - 正则化技术（L1, L2）与过拟合、欠拟合（如后续笔记本有详细内容可继续补充）
  - 相关代码实现与图形展示

### Week 4: 神经网络 (1) (Neural Networks Part 1)
- **文件**: `week4-NeuralNetworks(1).ipynb`
- **内容**:
  - 神经网络基础概念与结构（输入层、隐藏层、输出层）
  - 线性/逻辑回归的局限性与神经网络的优势
  - 前向传播公式推导与激活函数（sigmoid）作用
  - 多层感知机（MLP）结构图与数学表达
  - 逻辑函数（AND/OR）在神经网络中的实现与代码示例
  - 多分类输出的扩展（如softmax）
  - 相关代码实现与可视化

### Week 5: 神经网络 (2) (Neural Networks Part 2)
- **文件**: `week5-NeuralNetworks(2).ipynb`
- **内容**:
  - 神经网络的成本函数（如交叉熵损失、均方误差等）及其推导
  - 反向传播算法原理与数学推导（链式法则、梯度计算）
  - 反向传播的代码实现与梯度检查方法（数值梯度与解析梯度对比）
  - 参数初始化策略（如随机初始化、避免对称性）
  - 神经网络训练技巧（如正则化、学习率调整、mini-batch、早停法等）
  - 相关公式推导、代码实现与可视化

### Week 6: 应用、学习曲线与精确率召回率 (Applications, Learning Curve & PR)
- **文件**: `week6-Applications,LearningCurve&PR.ipynb`
- **内容**:
  - 典型机器学习应用案例（如垃圾邮件分类、疾病预测、图像识别等）及其数据特征
  - 学习曲线分析：训练集/验证集误差随样本量变化的趋势，诊断高偏差（欠拟合）与高方差（过拟合）
  - 精确率-召回率（Precision-Recall）曲线及其意义，PR曲线与ROC曲线的区别与适用场景
  - 主要模型评估指标：准确率（Accuracy）、精确率（Precision）、召回率（Recall）、F1分数等，及其计算公式
  - 交叉验证（Cross Validation）方法及其在模型选择与调参中的应用
  - 相关代码实现与可视化

### Week 7: 支持向量机 (Support Vector Machines)
- **文件**: `week7-SupportVectorMachines.ipynb`
- **内容**:
  - 支持向量机（SVM）基本原理：最大间隔分类思想与几何解释
  - 线性可分与不可分情况的处理方法
  - 核函数原理与常见类型（线性核、多项式核、RBF高斯核等），核技巧的作用
  - 软间隔与硬间隔的概念及其在实际问题中的应用
  - SVM参数调优（如C、gamma等）及其对模型性能的影响
  - 主要公式推导与支持向量的几何意义
  - 相关代码实现与可视化

### Week 8: 聚类与降维 (Clustering & Dimensionality Reduction)
- **文件**: `week8-Clustering&DimensionalityReduction.ipynb`
- **内容**:
  - K-means聚类算法原理、迭代流程（初始化、分配、更新中心）与收敛性分析
  - 层次聚类（Hierarchical Clustering）思想与常见方法（自底向上/自顶向下）
  - 主成分分析（PCA）数学原理（方差最大化、特征值分解）与降维流程
  - 奇异值分解（SVD）在数据降维与特征提取中的应用
  - 聚类与降维结果的可视化
  - 典型代码实现与实际应用场景

### Week 9: 异常检测与推荐系统 (Anomaly Detection & Recommender Systems)
- **文件**: `week9-AnomalyDetection&RecommenderSystems.ipynb`
- **内容**:
  - 异常检测算法原理（如基于概率的异常检测、密度估计等）及其应用场景（如欺诈检测、设备故障预警）
  - 高斯分布模型在异常检测中的建模与判别流程，参数估计与多变量高斯分布
  - 协同过滤（Collaborative Filtering）推荐原理：基于用户-物品评分矩阵的相似性计算
  - 内容推荐（Content-based Recommendation）方法与特征工程
  - 典型代码实现与实际案例分析

### Week 10: 大规模机器学习 (Large Scale Machine Learning)
- **文件**: `week10-LargeScaleMachineLearning.ipynb`
- **内容**:
  - 大规模机器学习的核心思想与挑战（如数据量大、计算资源有限、模型复杂度高）
  - 批量梯度下降、随机梯度下降、小批量梯度下降的原理与对比
  - 大数据处理策略：数据分片、并行计算、分布式存储
  - 在线学习与增量学习原理，适用于数据流场景的模型更新方法

## 参考链接：
[fengdu78机器学习个人笔记-目录](http://www.ai-start.com/ml2014/)
[Coursera课程视频](https://www.coursera.org/learn/machine-learning)