#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Alpha Labs



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Data_preprocessing_function import *
from sklearn.preprocessing import LabelEncoder

#读取文件
def read(df_1):
    print(df_1)
    #print(df_1.head(5))
    #print(df_1.describe())
    # print("%d行 %d列" % (df_1.shape[0], df_1.shape[1]))
    # print("每列缺失值的个数：", df_1.isnull().sum()) #每列缺失值的个数

#处理缺失数据
def processing_NA(df):
    df['Embarked'] = df['Embarked'].dropna()
    df = df.fillna(method='pad') #用前一个数值来填充
    return df

#处理字符串
def fun1(a):
    if a == 'male':
        return 1
    else :return 0

def processing_string(df):
    lst = list(map(fun1, df['Sex']))
    S_lst = pd.Series(lst)
    df['Sex'] = S_lst
    #用机器学习的函数 LableEncoder 对字符串进行编码
    le = LabelEncoder()
    le.fit(np.unique(df['Embarked'].values))
    df['Embarked'] = le.transform(df['Embarked'])
    return df
