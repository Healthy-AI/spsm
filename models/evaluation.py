import math

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from scipy.stats import norm
import pandas as pd
import numpy as np
from math import sqrt

import statistics


class Evaluation:
    def auc_ci(auc, n0, n1, alpha=0.05):
        q0 = auc / (2 - auc)
        q1 = 2 * auc * auc / (1 + auc)
        se = np.sqrt((auc * (1 - auc) + (n0 - 1) * (q0 - auc * auc) + (n1 - 1) * (q1 - auc * auc)) / (n0 * n1))

        z = norm.ppf(1 - alpha / 2)
        ci = z * se

        return (auc - ci, auc + ci)


    # Predict and evaluate
    def predict_and_evaluate(X_test, y_test, estimator, classification,S, I, label):
        X_test.loc[:, :] = I.transform(X_test)  # replace values of dataframe X_test with array values from imputer
        if classification == True:
            #accuracy and CI
            acc = accuracy_score(y_test, estimator.predict(S.transform(X_test)))
            n = np.sum(y_test)
            interval = 1.96 * sqrt((acc * (1 - acc)) / n)
            acc_l = acc - interval
            acc_u = acc + interval

            # auc and CI
            pp = estimator.predict_proba(S.transform(X_test))
            auc = roc_auc_score(y_test, pp[:, 1])

            if isinstance(y_test, pd.DataFrame):
                y_test = y_test.values

            n_y_test_0 = np.sum(y_test == y_test.min())
            n_y_test_1 = np.sum(y_test == y_test.max())
            ci_0, ci_1 = Evaluation.auc_ci(auc, n_y_test_0,  n_y_test_1, alpha=0.05)

            results = {'acc_' + label: [acc], 'acc_l_' + label: [acc_l], 'acc_u_' + label: [acc_u], 'auc_' + label: [auc], 'auc_l_' + label:[ci_0], 'auc_u_' + label:[ci_1]}
        else:
            #mse and CI
            mse = mean_squared_error(y_test, estimator.predict(S.transform(X_test)))
            #mse = mean_squared_error(y_test, estimator.predict(X_test))
            n = y_test.shape[0]
            interval = 1.96 * sqrt((2 * mse) / n)
            mse_l = mse - interval
            mse_u = mse + interval

            # r^2 and CI
            r_2 = r2_score(y_test, estimator.predict(S.transform(X_test)))
            #r_2 = r2_score(y_test, estimator.predict(X_test))

            # k = np.sum(X_test.columns)
            n= y_test.shape[0]
            interval = 1.96 * sqrt((1 - r_2) /(n -2))
            r_2_l = r_2 - interval
            r_2_u = r_2 + interval

            results = {'mse_' + label: [mse], 'mse_l_' + label: [mse_l], 'mse_u_' + label: [mse_u],'r_2_' + label: [r_2], 'r_2_l_' + label: [r_2_l], 'r_2_u_' + label: [r_2_u]}

        return results
