#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Alpha Labs

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
#使用kNN模型

def min_max(df):
    scaler = MinMaxScaler()
    scaler.fit(df[['Age', 'Fare', 'Embarked', 'Pclass']])
    df[['Age', 'Fare', 'Embarked', 'Pclass']] = scaler.transform(df[['Age', 'Fare', 'Embarked', 'Pclass']])
    return df

def KNeighborsClassifier_test(df):
    x_train, x_test, y_train, y_test = train_test_split(df.loc[:,'Pclass' : 'Embarked'], df['Survived'], test_size=0.25, random_state=50)
    clf = KNeighborsClassifier(n_neighbors=11)
    clf.fit(x_train, y_train)
    print("Training Score :", clf.score(x_train, y_train))
    print("Testing  Score :", clf.score(x_test,   y_test))
    print(clf.n_neighbors)

def test_KneighborsClassifier(df):
    x_train, x_test, y_train, y_test = train_test_split(df.loc[:,'Pclass' : 'Embarked'], df['Survived'], test_size=0.25, random_state=50)
    Ks = np.linspace(1, y_train.size / 10, endpoint=False, dtype='int')
    weights = ['uniform', 'distance']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for weight in weights:
        Test_scores = []
        Train_scores = []
        for K in Ks:
            clf = KNeighborsClassifier(weights=weight, n_neighbors=K)
            clf.fit(x_train, y_train)
            Test_scores.append(clf.score(x_test, y_test))
            Train_scores.append(clf.score(x_train, y_train))
        ax.plot(Ks, Test_scores, label='Test_scores : weight = %s' % weight)
        ax.plot(Ks, Train_scores, label='Train_scores : weight = %s' % weight)
    ax.legend(loc = 'best')
    ax.set_xlabel('K')
    ax.set_ylabel('score')
    ax.set_ylim(0.6, 1.05)
    ax.set_title("KNeighborsClassifier")
    plt.show()

def test_KNeighborsClassifier_k_p(df):
    x_train, x_test, y_train, y_test = train_test_split(df.loc[:,'Pclass' : 'Embarked'], df['Survived'], test_size=0.25, random_state=50)
    Ks = np.linspace(1, y_train.size / 10, endpoint=True, dtype='int')
    Ps = [1, 2, 10]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for p in Ps:
        train_scores = []
        test_scores = []
        for K in Ks:
            clf = KNeighborsClassifier(p=p, n_neighbors=K)
            clf.fit(x_train, y_train)
            train_scores.append(clf.score(x_train, y_train))
            test_scores.append(clf.score(x_test, y_test))
        ax.plot(Ks, train_scores, label='train_scores : p=%d'%p)
        ax.plot(Ks, test_scores, label='test_scores : p=%d'%p)
        ax.legend(loc='best')
        ax.set_xlabel('K')
        ax.set_ylabel('score')
        ax.set_ylim(0.6, 1.05)
        ax.set_title('KNeighborsClassifier')
    plt.show()

def run_main():
    df = pd.read_csv('./preprocessed_data.csv')
    print(df.head())
    df = min_max(df)
    KNeighborsClassifier_test(df)
    # test_KneighborsClassifier(df)
    test_KNeighborsClassifier_k_p(df)

if __name__ == '__main__':
    run_main()
