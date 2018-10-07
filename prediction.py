#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Alpha Labs


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier

#查看数据的相关性,选取特征
def Correlation(df):
    df_1 = df.corr()
    sns.heatmap(df_1, annot=True) #annot 显示相关性的大小
    plt.show()

#用分类决策树模型
from sklearn.model_selection import train_test_split
def DecisionTreeClassifier_test(df):
    x_train, x_test, y_train, y_test = train_test_split(df.ix[:, 'Pclass':'Embarked'],df['Survived'], test_size=0.25, random_state=50)
    # #找合适的max_depth参数  7
    # lst = list(range(1, 20))
    # test_scores = []
    # train_scores = []
    # for i in lst:
    #     clf = DecisionTreeClassifier(max_depth=i)
    #     clf.fit(x_train, y_train)
    #     test_scores.append(clf.score(x_test, y_test))
    #     train_scores.append(clf.score(x_train, y_train))
    # fig = plt.figure()
    # ax = fig.add_subplot(1,1,1)
    # ax.plot(lst, test_scores, label= 'test_scores', marker='o', c='red')
    # ax.plot(lst, train_scores, label='train_scores', marker='*', c='blue')
    # ax.set_xlabel("tree_deep")
    # ax.set_ylabel('score')
    # ax.set_title('DecisionTreeClassifier')
    # ax.legend(loc="best", framealpha=0.5)
    # plt.savefig('./DecisionTreeClassifier.png')
    # plt.show()

    X = pd.read_csv('./preprocessed_test_data.csv')
    clf = DecisionTreeClassifier(max_depth=7) #test_score: 0.807174887892  train_score : 0.899700598802
    clf.fit(x_train, y_train)
    pre = clf.predict(X)
    S1 = pd.Series(pre, name='Survived')
    df1 = pd.read_csv('./gender_submission.csv')
    df2 = pd.concat([df1['PassengerId'],S1], axis=1)
    # print('train_score :', clf.score(x_train, y_train))
    # print('test_score:', clf.score(x_test, y_test))
    # print('tree_deep:', clf.max_depth)
    # print(df2.head())
    df2.to_csv('./result.csv', index='None')






def run_mian():
    df_1 = pd.read_csv('./preprocessed_data.csv')
    Correlation(df_1)  #查看数据的相关性,选取特征
    DecisionTreeClassifier_test(df_1)


if __name__ == '__main__':
    run_mian()
