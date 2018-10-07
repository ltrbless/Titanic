#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Alpha Labs



import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler

# def preprocessing_encoder(df):
#     encoder = OneHotEncoder(sparse=False)
#     encoder.fit(df.loc[:,['SibSp', 'Embarked']].values)
#
#     # print(df.loc[:,'SibSp'].values.reshape(-1,1 ))
#     # print(df.loc[:,'SibSp'].values)
#
#     # print(df.loc[:, ['SibSp','Embarked']].values)
#     df2 = encoder.transform(df.loc[:, ['SibSp', 'Embarked']].values)
#
#     # print(df2.shape)
#     # print(df2)
#
#     df3 = df.values
#
#     df_all = np.hstack((df2, df3))
#
#     return df_all
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier
import xgboost
def run_main():
    df_train = pd.read_csv('./preprocessed_data.csv')
    df_test = pd.read_csv('./preprocessed_test_data.csv')
    X = np.array(df_train.drop(['Survived'], 1))
    y = np.array(df_train['Survived'])
    # print(X,y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    tpot = TPOTClassifier(generations=20, verbosity=2)
    tpot.fit(X_train, y_train)
    tpot.score(X_test, y_test)#0.824626865672
    tpot.export('tpot_exported_pipeline.py')
Best pipeline: RandomForestClassifier(RobustScaler(input_matrix), bootstrap=True, criterion=gini, max_features=0.8, min_samples_leaf=4, min_samples_split=15, n_estimators=100)

Optimization Progress: 100%|██████████| 2100/2100 [2:30:47<00:00,  1.24s/pipeline]Generation 20 - Current best internal CV score: 0.828215053763441



if __name__ == '__main__':
    run_main()















