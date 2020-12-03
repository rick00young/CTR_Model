# -*- coding: utf-8 -*-
# @Time    : 12/3/20 15:50
# @Author  : yangyongrui
# @File    : fm.py


import numpy as np
from sklearn.feature_extraction import DictVectorizer
from pyfm import pylibfm
import pandas as pd
from sklearn.metrics import accuracy_score, auc, roc_auc_score
from sklearn.model_selection import train_test_split

# Read in data
def loadRatingData(filename,path="data/ml-100k/"):
    data = []
    y = []
    users=set()
    items=set()
    with open(path+filename) as f:
        for line in f:
            (user,movieid,rating,ts)=line.split('\t')
            data.append({"user_id": str(user), "movie_id": str(movieid)})
            y.append(float(rating))
            users.add(user)
            items.add(movieid)

    return (data, np.array(y), users, items)

def fm_rating_model():
    (train_data, y_train, train_users, train_items) = loadRatingData("ua.base")
    (test_data, y_test, test_users, test_items) = loadRatingData("ua.test")
    v = DictVectorizer()
    X_train = v.fit_transform(train_data)
    X_test = v.transform(test_data)

    # Build and train a Factorization Machine
    fm = pylibfm.FM(num_factors=10, num_iter=100, verbose=True, task="regression", initial_learning_rate=0.001, learning_rate_schedule="optimal")

    fm.fit(X_train,y_train)
    # Evaluate
    preds = fm.predict(X_test)
    from sklearn.metrics import mean_squared_error
    print("FM MSE: %.4f" % mean_squared_error(y_test, preds))
    # FM MSE: 0.8873

#-------------

def load_gbdt_data():
    path = '../GBDT_lR/data/'
    df_train = pd.read_csv(path + 'train.csv')
    df_test = pd.read_csv(path + 'test.csv')
    df_train.drop(['Id'], axis=1, inplace=True)
    df_test.drop(['Id'], axis=1, inplace=True)
    df_test['Label'] = -1
    data = pd.concat([df_train, df_test])
    data = data.fillna(-1)
    data.to_csv('data/data.csv', index=False)
    return data

def fm_gbdt_model():
    data = load_gbdt_data()
    continuous_feature = ['I'] * 13
    continuous_feature = [col + str(i + 1) for i, col in enumerate(continuous_feature)]
    category_feature = ['C'] * 26
    category_feature = [col + str(i + 1) for i, col in enumerate(category_feature)]

    # 类别特征one-hot编码
    for col in category_feature:
        onehot_feats = pd.get_dummies(data[col], prefix=col)
        data.drop([col], axis=1, inplace=True)
        data = pd.concat([data, onehot_feats], axis=1)

    train = data[data['Label'] != -1]
    target = train.pop('Label')
    test = data[data['Label'] == -1]

    train_data = []
    for i in range(len(train)):
        train_data.append(train.loc[i].to_dict())

    x_train, x_val, y_train, y_val = train_test_split(train_data, target, test_size=0.2, random_state=2020)

    v = DictVectorizer()

    X_train = v.fit_transform(x_train)

    X_val = v.transform(x_val)

    fm = pylibfm.FM(num_factors=50, num_iter=5, verbose=True, task="classification", initial_learning_rate=0.0001,
                    learning_rate_schedule="optimal")

    y_predict = fm.fit(X_train, y_train)

    auc_score = roc_auc_score(y_predict, y_val)
