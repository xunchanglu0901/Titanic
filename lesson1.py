# -*- coding: utf-8 -*-
# 时间    : 2019/1/14 19:35
# 作者    : xcl

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
'''
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
'''
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

