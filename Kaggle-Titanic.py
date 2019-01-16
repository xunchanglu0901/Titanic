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
img=Image.open('models.png')
img.show()
'''
img=Image.open('models.png')
plt.figure('models')
plt.imshow(img)
plt.show()