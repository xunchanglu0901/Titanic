# -*- coding: utf-8 -*-
# 时间    : 2019/1/15 18:55
# 作者    : xcl



###### 第一堂课 ######

'''
import warnings
warnings.filterwarnings('ignore')
#载入数据
import pandas as pd
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv("test.csv")
#查看数据格式
#print(train_df.shape,test_df.shape)
import  matplotlib.pyplot as plt
#查看列名,部分数据
#print(train_df.columns.values)
#print(test_df.head())
#内容信息
#print(train_df.info())
#缺失值
#print(train_df.isnull().sum())
#基本描述
#print(test_df.describe())

#绘图

plt.hist(test_df['PassengerId'])
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()
'''
'''
plt.hist([train_df[train_df["Sex"]=="male"],Age,train_df[train_df["Sex"]=="female"],Age])
plt.legend(['male','female'])
plt.xlabel('Age')
plt.ylabel('Count')

crosstab1 = pd.crosstab(train_df['Survived'],train_df["Sex"])
print(crosstab1)
group1 = train_df.groupby(["Embarked","Survived"]).size()
print(group1)

from pandasql import sqldf
ql = """
SELECT Survived,Parch,count(*)
FROM train_df
GROUP BY Survived,Parch
"""
ql = sqldf(ql)
print(ql)
'''



###### 第二堂课 ######

import warnings
warnings.filterwarnings('ignore')
#载入数据
import pandas as pd
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv("test.csv")
#载入处理数据框的包
import numpy as np
train_df.tail() #查看末尾几行
#处理缺失值：进行填充
train_df.isnull().sum()#缺失值个数，Cabin缺失过多
train_df.Embarked = train_df["Embarked"].fillna(method="ffill")#使用前一个值进行填充
s = train_df["Age"].value_counts(normalize=True)#各年龄层百分比
missing_age=train_df["Age"].isnull()#判断是否为空
train_df.head(10)
#随机填充，根据数据的年龄层百分比，填充后百分比一致
train_df.loc[missing_age,"Age"] = np.random.choice(s.index,size=len(train_df[missing_age]),p=s.values)
#print(train_df["Age"].head(60))#查看填充结果
train_df = train_df.drop(['Ticket','Cabin'], axis = 1)#删除无用行列
#正则表达式提取称呼
train_df['Title'] = train_df.Name.str.extract(' ([A-Za-z]+)\.',expand = False)
train_df.Title.unique()
pd.crosstab(train_df['Title'], train_df['Survived'])# 依据乘客的称呼和最后的生存情况创建交叉表
# 替换称呼来重新赋值Title,如将Lady Countess等出现频次较低的重新命名为Rare
train_df['Title'] = train_df['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')#长句换行，使用\
train_df['Title'] = train_df['Title'].replace('Mlle', 'Miss')
train_df['Title'] = train_df['Title'].replace('Ms', 'Miss')
train_df['Title'] = train_df['Title'].replace('Mme', 'Mrs')#replace，后替换前
# 通过groupby进行分组，选取Title 和 Survived两列，然后使用mean()计算每一种Title的乘客的生还率
Survived_rate = train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
#print(Survived_rate)
# 定义一个字典title_mapping，将原来Title中的值一次映射为字典里对应的值
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}#类似虚拟变量
train_df['Title'] = train_df['Title'].map(title_mapping)
# 用0来填充缺失值
train_df['Title'] = train_df['Title'].fillna(0)
#创建新变量，家族成员数
train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
train_df['IsAlone'] = 1
train_df['IsAlone'].loc[train_df['FamilySize'] > 1] = 0
# 依据票价绘制直方图
import matplotlib.pyplot as plt
#plt.hist(train_df['Fare'])
#plt.show()

#利用cut把数据用25%，50%，75%进行划分
train_df['FareBin'] = pd.qcut(train_df['Fare'], 4)
#利用cut把数据分成均等的五份
train_df['AgeBin'] = pd.cut(train_df['Age'], 5)
# 调用机器学习库scikit-learn中的LabelEncoder
from sklearn.preprocessing import LabelEncoder
# 创建一个名为label_encode的LabelEncoder对象
label_encode = LabelEncoder()
# 有下列label
labels = ['Sex','Embarked','AgeBin','FareBin']
# 遍历labels列表
for label in labels:
    # 输出每一个label和它的数据类型
    #print (label,type(label))
    # 新的标签名在原标签名的基础上加上_Code
    new_label = label + '_Code'
    # 调用LabelEncoder的fit_transform方法对原train_df[label]先拟合再标准化
    train_df[new_label] = label_encode.fit_transform(train_df[label])
original_feature = ['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
new_feature = ['Title','FamilySize','IsAlone','Sex_Code','Embarked_Code','AgeBin_Code','FareBin_Code']
'''
上述操作完成了，字符串变量到数字型(离散、连续)变量的转化，从而便于应用于后续模型的拟合
'''
#载入seaborn包并以sns简化命名，当你使用数据科学中的Python时，你很有可能已经用了Matplotlib,一个供你创建高质量图像的2D库。
#另一个免费的可视化库就是Seaborn,他提供了一个绘制统计图形的高级接口。Seaborn是比Matplotlib更高级的免费库，特别地以数据可视化为目标，
#Matplotlib试着让简单的事情更加简单，困难的事情变得可能，而Seaborn就是让困难的东西更加简单。
#用Matplotlib最大的困难是其默认的各种参数，而Seaborn则完全避免了这一问题。
import seaborn as sns
# 在上面，将众多属性名聚在一个list命名为new_feature
# 此处使用seaborn的heatmap方法，画出new_feature中各种特征两两之间的相似度热力图
#sns.heatmap(train_df[new_feature].corr(),annot=True,cmap='cubehelix_r')
#plt.show()



###### 第三堂课 ######

# 从scikit-learn中引入OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
# 创建一个名为enc的OneHotEncoder对象
enc = OneHotEncoder()
# 创建一个名为onehot_features的list,并存放要进行编码的属性
onehot_features = ['Title','Sex_Code','Embarked_Code','AgeBin_Code','FareBin_Code']
# 对这些属性进行独热编码并且求得train_df[onehot_features]的均值等属性
enc.fit(train_df[onehot_features])
# 查看enc对象里各个分类编码过后的值
#enc.categories_
# 创建一个名为enc_res的对象存放经过标准化处理后的独热编码的值
enc_res = enc.transform(train_df[onehot_features])
# 查看enc_res的值的情况
#print(enc_res.toarray())
# 查看enc_res的数据维度
#print(enc_res.toarray().shape)
# 选取不同变量，创建两个List（分别为原先的特征List和新的特征List）来分别存放特征
original_features = ['PassengerId','Pclass', 'Name', 'Sex', 'Age' ,'SibSp', 'Parch','Ticket','Fare', 'Cabin', 'Embarked']
new_features = ['Title','FamilySize','IsAlone','Sex_Code','Embarked_Code','AgeBin_Code','FareBin_Code']

final_features = ['Pclass','Age','SibSp','Parch','Fare','Title','FamilySize',
                  'IsAlone','Sex_Code','Embarked_Code','AgeBin_Code','FareBin_Code']
# 删去难以运用的变量&重复的变量
onehot_final = list(set(final_features) - set(onehot_features))
#打印出onehot_final变量的名称
#print(onehot_final)
# 将原来数据里的final_features这些列存放到all_x里方便后面使用
all_x = train_df[final_features]
# 引入原先数据'survive'并且赋值y
y = train_df['Survived']
# 查看all_x前几行的信息
all_x.head()
# 将转换为数组的独热编码的值存放到onehot_added里
onehot_added = pd.DataFrame(enc_res.toarray())
# 使用pandas的concat函数将原数据中onehot_final这些列和onehot_added合并起来，concat函数专门用于连接两个或多个数组
# axis指定了合并的轴，此处axis=1意为逐列合并，若axis=0则为逐行合并；合并后的函数赋值为新的数据集all_x_2
all_x_2 = pd.concat([train_df[onehot_final],onehot_added],axis = 1)
# 查看all_x_2前几行的信息
#print (all_x_2.head())
#确认一下没有缺失值
#print(all_x.isnull().sum())
# 从scikit-learn中引入train_test_split
from sklearn.model_selection import train_test_split
# 对all_x进行数据集划分为训练集和测试集：xTrain为训练集数据，xTest为测试集数据
# y为数据集的标签（即该乘客是否存活），yTrain对应了训练集的标签，yTest对应了测试集的标签
# test_size=0.2表示测试集占总数据集的20%
xTrain, xTest, yTrain, yTest = train_test_split(all_x, y, test_size = 0.2, random_state = 0)
# 查看训练集和测试集的数据量
#print(xTrain.shape, xTest.shape)
# 查看训练集标签数和测试集标签数
#print(yTrain.shape,yTest.shape)
# 计算训练集中乘客存活率平均值和测试集中乘客存活率平均值
#print(yTrain.mean(),yTest.mean())
# 同样的对all_进行数据集划分
x2Train, x2Test, y2Train, y2Test = train_test_split(all_x_2, y, test_size = 0.2, random_state = 0)
# 如何选择适合的机器学习模型 ★
#pip install Pillow
from PIL import Image
'''
#method one
img=Image.open('models.png')
img.show()
#method two
img=Image.open('models.png')
plt.figure('models')
plt.imshow(img)
plt.show()
'''



###### 第四堂课 ######

#从sklearn中调用逻辑回归、决策树，随机森林包
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
#初始化模型，分别存入logr,dtree,rf变量中，并将变量置于名为models的列表中
logr = LogisticRegression()
dtree = DecisionTreeClassifier()
rf = RandomForestClassifier()
models = [logr,dtree,rf]
#这个函数调用逻辑回归模型来学习训练集和测试集
logr.fit(xTrain,yTrain)
#根据自变量xTest,xTrain分别利用训练好的模型预测出因变量
y_pred_test = logr.predict(xTest)
y_pred_train = logr.predict(xTrain)
#算出测试集精确度（求均值）
np.mean(y_pred_test == yTest)
#算出训练集精确度
np.mean(y_pred_train == yTrain)
#对每一个模型，分别测试训练集和测试集的精确度
for model in models:
    #print ('\nThe current model is', model)
    model.fit(xTrain, yTrain)
    #print ('\nTraining accuracy is',np.mean(model.predict(xTrain) == yTrain))
    #print ('\nTesting accuracy is',np.mean(model.predict(xTest) == yTest))
#对第二组数据进行相同的操作，每一个模型，分别测试训练集和测试集的精确度
for model in models:
    #rint ('\nThe current model is', model)
    model.fit(x2Train, y2Train)
    #print ('\nTraining accuracy is',np.mean(model.predict(x2Train) == y2Train))
    #print ('\nTesting accuracy is',np.mean(model.predict(x2Test) == y2Test))

#交叉验证是机器学习领域常用的验证模型是否优秀的方法；简而言之就是把数据切分成几个部分然后在训练集和测试集中交换使用
#比如这次在训练集中用到的数据，下一次会放进测试集来使用，因此被称为 交叉
#简单的交叉验证会直接按百分比来划分训练集和测试集，更难一些的方法是K-Fold，我们会使用这个方法来进行交叉验证
#K-Fold其实就是把数据集切成K份，将每一份儿小数据集分别做一次验证集，每次剩下的K-1组份儿数据作为训练集，得到K个模型
#
#从sklearn中调用k次交叉验证包
from sklearn.model_selection import KFold


# 定义k次交叉验证函数
def CVKFold(k, X, y, Model):
    # Random seed: reproducibility
    np.random.seed(1)

    # accuracy score
    train_accuracy = [0 for i in range(k)]
    test_accuracy = [0 for i in range(k)]

    # index
    idx = 0

    # CV loop
    kf = KFold(n_splits=k, shuffle=True)

    # Generate the sets
    for train_index, test_index in kf.split(X):
        # Iteration number
        # print(train_index,len(train_index))
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Calling the function/model

        if Model == "Logit":
            clf = LogisticRegression(random_state=0)

        if Model == "RForest":
            clf = RandomForestClassifier(random_state=0)

        # Fit the model
        clf = clf.fit(X_train, y_train)
        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)

        train_accuracy[idx] = np.mean(y_train_pred == y_train)
        test_accuracy[idx] = np.mean(y_test_pred == y_test)
        idx += 1

    #print(train_accuracy)
    #print(test_accuracy)
    return train_accuracy, test_accuracy
#应用逻辑回归模型，将数据分为10份，每次取一个样本作为验证数据，剩下k-1个样本作为训练数据
train_acc,test_acc = CVKFold(10,all_x,y,"Logit")
#验证训练数据和测试数据的精确度
np.mean(train_acc),np.mean(test_acc)
#应用随机森林模型，将数据分为10份，每次取一个样本作为验证数据，剩下k-1个样本作为训练数据
train_acc,test_acc = CVKFold(10,all_x,y,"RForest")



###### 第五堂课 ######

#查看数据信息
xTrain.head()
xTrain.shape
#分别创建名为logr，dtree，rf的逻辑回归，决策树和随机森林对象
logr = LogisticRegression()
dtree = DecisionTreeClassifier()
rf = RandomForestClassifier()
#调用LogisticRegression的fit方法对数据集进行拟合，C为正则化系数λ的倒数，通常默认为1，class_weight参数用于标示分类模型中各种类型的权重，
#dual：一个布尔值，如果为true，则求解对偶形式（只在penalty=‘l2’且solver=‘lib-linear’有对偶形式），如果为false，则求解原始形式，
#默认为false；fit_intercept，是否存在截距，默认存在；intercept_scaling:一个浮点数，只有当solver='liblinear'才有意义。当采用
#fit_intcept时相当于人造一个特征出来，特征恒为1，权重为b。在计算正则化项的时候，该人造特征也被考虑了，因此为了降低这个人造特征的影响，
#需要提供intercept_scaling；max_iter:一个整数，指定最大迭代次数；multi_class:一个字符串，指定对于多分类问题的策略，可以为如下值，
#ovr:采用one-vs-rest策略；multinomial:直接采用多分类逻辑回归策略。n_jobs：一个正数。指定任务并行时的cpu数量。如果为-1则使用所有可用的CPU。
#random_state:一个整数或者一个RandomState实例或者none。
#如果为整数，则它指定了随机数生成器的种子，如果为RandomState实例，则指定了随机数生成器。如果为none则使用默认是随机数生成器。
#solver:一个字符串，指定了求解最优化问题的算法，可以为如下值：newton-cg:使用牛顿法；lbfgs:使用L-BFGS拟牛顿法；
#liblinear:使用liblinear;sag:使用stochastic average gradient descent算法。注：对于规模小的数据集，liblinear比较适用，对于规模大的数据集，
#sag比较适用，而newton-cg,lbfgs、sag只处理penalty='l2'的情况。tol:一个浮点数，指定判断迭代收敛与否的阈值。
#verbose:一个正数。用于开启/关闭迭代中间输出日志功能。warm_start:一个布尔值，如果为true，那么使用前一次训练结果继续训练，否则从头开始训练。
logr.fit(xTrain,yTrain)
#查看拟合后logistic regression模型的intercept值
logr.intercept_
#查看拟合后logistic regression模型的coef值
logr.coef_
#打开名为“logrformular.png“的图像
'''
img=Image.open('logrformular.png')
plt.figure('logrformular')
plt.imshow(img)
plt.show()
'''
#载入graphviz数据包，以及从sklearn中载入tree数据包：graphviz是AT&T实验室开源的画图工具，
import graphviz
from sklearn import tree
#创建名为dtree的决策树对象
dtree = DecisionTreeClassifier()
#调用DecisionTree的fit方法对数据集进行拟合
#class_weight:dict,list of dicts,"Banlanced" or None,可选（默认为None）
#criterion:string类型，可选（默认为"gini"）衡量分类的质量。支持的标准有："gini"代表的是Gini impurity(不纯度)；
#"entropy"代表的是information gain（信息增益）。
#max_depth:int or None,可选（默认为"None"）表示树的最大深度。如果是"None",则节点会一直扩展直到所有的叶子都是纯的或者所有的叶子节点都包含
#少于min_samples_split个样本点。忽视max_leaf_nodes是不是为None。
#max_features:int,float,string or None 可选（默认为None），在进行分类时需要考虑的特征数。None，max_features=n_features 注意：至少找到一个
#样本点有效的被分类时，搜索分类才会停止。
#max_leaf_nodes:int,None 可选（默认为None）在最优方法中使用max_leaf_nodes构建一个树。最好的节点是在杂质相对减少。
#如果是None则对叶节点的数目没有限制。如果不是None则不考虑max_depth.
#min_impurity_decrease : float, optional (default=0.)如果该分裂导致杂质的减少大于或等于该值，则将分裂节点。
#min_samples_leaf:int,float,可选（默认为1）一个叶节点所需要的最小样本数。
#min_samples_split:int,float,可选（默认为2）区分一个内部节点需要的最少的样本数。
#min_weight_fraction_leaf:float,可选（默认为0）一个叶节点的输入样本所需要的最小的加权分数。
#persort:bool,可选（默认为False）是否预分类数据以加速训练时最好分类的查找。在有大数据集的决策树中，如果设为true可能会减慢训练的过程。
#当使用一个小数据集或者一个深度受限的决策树中，可以减速训练的过程。
#random_state:int,RandomState instance or None；如果是None，随机数字发生器是np.random使用的RandomState instance.
#splitter:string类型，可选（默认为"best"） 一种用来在节点中选择分类的策略。支持的策略有"best"，选择最好的分类，"random"选择最好的随机分类。
dtree.fit(xTrain,yTrain)
#利用tree的export_graphviz以DOT形式提取决策树
dot_data = tree.export_graphviz(dtree, out_file=None,feature_names=xTrain.columns,filled=True, rounded=True)
#将DOT形式的决策树源码字符串形式
graph = graphviz.Source(dot_data)
#用pdf存储源码
graph.render('dtree')
#创建名为rf_10的随机森林对象，森林里决策树的数目为10
rf_10 = RandomForestClassifier(n_estimators=10)
#调用RandomForestClassifier的fit方法对数据集进行拟合
#bootstrap=True：是否有放回的采样。
#max_features: 选择最适属性时划分的特征不能超过此值。当为整数时，即最大特征数；当为小数时，训练集特征数*小数；
#if “auto”, then max_features=sqrt(n_features).
#n_estimators=10：决策树的个数，越多越好，但是性能就会越差，至少100左右
#n_jobs=1：并行job个数。这个在ensemble算法中非常重要，尤其是bagging（而非boosting，因为boosting的每次迭代之间有影响，所以很难进行并行化），
#因为可以并行从而提高性能。1=不并行；n：n个并行；-1：CPU有多少core，就启动多少job。
#oob_score=False：oob（out of band，带外）数据，即：在某次决策树训练中没有被bootstrap选中的数据。
#verbose:(default=0) 是否显示任务进程
#warm_start=False：热启动，决定是否使用上次调用该类的结果然后增加新的。
rf_10.fit(xTrain,yTrain)
#查看刚才创建的决策树中的第五棵树
rf_5 = rf_10.estimators_[5]
#利用tree的export_graphviz以DOT形式提取决策树
dot_data = tree.export_graphviz(rf_5, out_file=None,feature_names=xTrain.columns,filled=True, rounded=True)
#将DOT形式的决策树源码字符串形式
graph = graphviz.Source(dot_data)
#用pdf存储源码
graph.render('rftree')
#无论在机器学习还是深度学习建模当中都可能会遇到两种最常见结果，一种叫过拟合（over-fitting ）另外一种叫欠拟合（under-fitting）。
#所谓过拟合（over-fitting）其实就是所建的机器学习模型或者是深度学习模型在训练样本中表现得过于优越，导致在验证数据集以及测试数据集中表现不佳。
#过拟合就是学到了很多没必要的特征。
#导入图像，诠释什么是过度拟合
'''
img=Image.open('overfit.png')
plt.figure('overfit')
plt.imshow(img)
plt.show()
'''
#利用decisiontree的predict预测测试集和训练集
y_pred_train = dtree.predict(xTrain)
y_pred_test = dtree.predict(xTest)
#利用numpy数据包的mean取平均，评估预测的精确度
np.mean(y_pred_train == yTrain),np.mean(y_pred_test == yTest)
#创建名为dtree2的决策树对象，规定树的最大深度为5，构成一个内部节点的样本最少为5个
dtree2 = DecisionTreeClassifier(max_depth=5,min_samples_split=5)
#利用DecisionTreeClassifier的fit方法拟合xy训练集
dtree2.fit(xTrain,yTrain)
#利用numpy数据包的mean取平均，评估预测的精确度
np.mean(dtree2.predict(xTrain) == yTrain),np.mean(dtree2.predict(xTest) == yTest)
print(np.mean(dtree2.predict(xTrain) == yTrain),np.mean(dtree2.predict(xTest) == yTest))