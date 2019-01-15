# -*- coding: utf-8 -*-
# 时间    : 2019/1/15 18:55
# 作者    : xcl

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
plt.hist(train_df['Fare'])
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
    print (label,type(label))
    # 新的标签名在原标签名的基础上加上_Code
    new_label = label + '_Code'
    # 调用LabelEncoder的fit_transform方法对原train_df[label]先拟合再标准化
    train_df[new_label] = label_encode.fit_transform(train_df[label])
original_feature = ['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
new_feature = ['Title','FamilySize','IsAlone','Sex_Code','Embarked_Code','AgeBin_Code','FareBin_Code']
#载入seaborn包并以sns简化命名，当你使用数据科学中的Python时，你很有可能已经用了Matplotlib,一个供你创建高质量图像的2D库。
#另一个免费的可视化库就是Seaborn,他提供了一个绘制统计图形的高级接口。Seaborn是比Matplotlib更高级的免费库，特别地以数据可视化为目标，
#Matplotlib试着让简单的事情更加简单，困难的事情变得可能，而Seaborn就是让困难的东西更加简单。
#用Matplotlib最大的困难是其默认的各种参数，而Seaborn则完全避免了这一问题。
import seaborn as sns
# 在上面，将众多属性名聚在一个list命名为new_feature
# 此处使用seaborn的heatmap方法，画出new_feature中各种特征两两之间的相似度热力图
sns.heatmap(train_df[new_feature].corr(),annot=True,cmap='cubehelix_r')
plt.show()