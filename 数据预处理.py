#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Alpha Labs


if __name__ == '__main__':
    import pandas as pd
    import numpy as ny
    import matplotlib.pyplot as plt
    from Data_preprocessing_function import *

    #读取文件
    filename = "./test.csv"
    df_1 = pd.read_csv(filename)
    #read(df_1)

    #删除没用的列
    df_1 = df_1.drop(['Name', 'Ticket', 'Cabin'], axis=1)
    #read(df_1)

    #处理缺失数据
    df_1 = processing_NA(df_1)
    #read(df_1)

    #处理字符串
    df_1 = processing_string(df_1)
    read(df_1)

    #把处理好的数据存起来
    df_1.to_csv('./preprocessed_test_data.csv')








