import pandas as pd
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import neighbors
from sklearn.svm import SVR

def alg_test(X_train, y_train):
    cv_k = 5
    cv_scoring = 'neg_mean_squared_error'
    kf = KFold(n_splits=cv_k, shuffle=True)
    score_all = {}

    # LR
    print('LR:')
    now = time.time()
    est = linear_model.LinearRegression()
    scores = cross_val_score(est, X_train, y_train, cv=kf, scoring=cv_scoring)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    after = time.time()
    print('Exec. time: {:5.2f} s'.format(after-now))
    score_all['LR'] = scores.mean()

    # LRR
    print('LRR')
    now = time.time()
    est = linear_model.Ridge(alpha = 1.0)
    scores = cross_val_score(est, X_train, y_train, cv=kf, scoring=cv_scoring)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    after = time.time()
    print('Exec. time: {:5.2f} s'.format(after-now))
    score_all['LRR'] = scores.mean()

    #
    # print("PR")
    # now = time.time()
    # poly = PolynomialFeatures(degree=2)
    # X_train_poly = poly.fit_transform(X_train)
    # est = linear_model.LinearRegression()
    # scores = cross_val_score(est, X_train_poly, y_train, cv=kf, scoring=cv_scoring)
    # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    # after = time.time()
    # print('Exec. time: {:5.2f} s'.format(after-now))
    # score_all['PR'] = scores.mean()

    # print("PRR")
    # now = time.time()
    # poly = PolynomialFeatures(degree=2)
    # X_train_poly = poly.fit_transform(X_train)
    # est = linear_model.Ridge(alpha = 1.0)
    # scores = cross_val_score(est, X_train_poly, y_train, cv=kf, scoring=cv_scoring)
    # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    # after = time.time()
    # print('Exec. time: {:5.2f} s'.format(after-now))
    # score_all['PRR'] = scores.mean()

    # RF
    print("RF:")
    now = time.time()
    est = RandomForestRegressor(n_estimators=10, n_jobs=-1)
    scores = cross_val_score(est, X_train, y_train, cv=kf, scoring=cv_scoring)
    print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
    after = time.time()
    print('Exec. time: {:5.2f} s'.format(after-now))
    score_all['RFR'] = scores.mean()

    #
    print("GBR")
    now = time.time()
    est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                    max_depth=0.3, random_state=0, loss='ls')
    scores = cross_val_score(est, X_train, y_train, cv=kf, scoring=cv_scoring)
    print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
    after = time.time()
    print('Exec. time: {:5.2f} s'.format(after-now))
    score_all['GBR'] = scores.mean()

    print("KNN")
    now = time.time()
    n_neighbors = 5
    weight = 'uniform' # 'distance'
    est = neighbors.KNeighborsRegressor(n_neighbors, weights=weight, n_jobs=-1)
    scores = cross_val_score(est, X_train, y_train, cv=kf, scoring=cv_scoring)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    after = time.time()
    print('Exec. time: {:5.2f} s'.format(after-now))
    score_all['KNN'] = scores.mean()

    score_all = pd.DataFrame({"score": score_all})
    score_all = score_all.sort_values('score', ascending=False)
    return score_all