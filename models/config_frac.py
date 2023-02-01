from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.linear_model import LinearRegression, Ridge
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor

from models import sharing_models
import sys

datasets = ['ADNI_cla','ADNI_reg', 'SUPPORT_cla', 'SUPPORT_reg']

def get_estimators():
    estimators = {
        'lr': LogisticRegression,
        'xgbc': XGBClassifier,
        'mlpc': MLPClassifier,
        'SM_lr': sharing_models.SharingLogisticSubModel,
        'PSM_lr': sharing_models.SharingLogisticSubModel,
        'SPSM_lr': sharing_models.SharingLogisticSubModel,
        'ols': LinearRegression,
        'ridge': Ridge,
        'xgbr': XGBRegressor,
        'mlpr': MLPRegressor,
        'SM_ols': sharing_models.SharingLinearSubModel,
        'PSM_ols': sharing_models.SharingLinearSubModel,
        'SPSM_ols': sharing_models.SharingLinearSubModel
    }
    return estimators


def get_config_ADNI_cla(): #function that returns dataset, estimator and when

    estimators= [get_estimators()[e] for e in ['lr', 'xgbc', 'mlpc','SM_lr', 'PSM_lr', 'SPSM_lr']]

    # dictionary with all parameter options for all logistic estimators
    parameters_all = {
    'lr': {'penalty': ['l1', 'l2'], 'fit_intercept': [True], 'i': ['mean', 'iterate', 'zero'], 'sp':[0.2], 's': [0,1,2,3,4], 'fr': [.2, .4, .6, .8, 1.]},
    #'xgbc': {'n_estimators': [100], 'learning_rate': [1.0], 'max_depth': [5, 10, 15], 'i': ['mean','iterate', 'zero', 'none'], 'sp':[0.2], 's': [0,1,2,3,4]},
    #'mlpc': {'hidden_layer_sizes': [10,20,30], 'activation': ['relu'], 'solver': ['adam'],
             #'alpha': [0.0001, 0.05], 'learning_rate': ['adaptive'],'i': ['mean','iterate', 'zero'], 'sp':[0.2], 's': [0,1,2,3,4]},
    'PSM_lr': {'alpha0':[1e8], 'alphap': [0], 'tol':[1e-5], 'penalty_main':['l2'],
                'min_samples_pattern':[40], 'reg_pattern_intercept':[False], 'i': ['none'], 'sp':[0.2], 's': [0,1,2,3,4], 'fr': [.2, .4, .6, .8, 1.]},
    'SPSM_lr': {'alpha0':[0, 0.1, 1, 10], 'alphap': [1, 50, 10, 100, 1000], 'tol':[1e-5], 'penalty_main':['l1'],
                'min_samples_pattern':[0, 40], 'reg_pattern_intercept':[False], 'i': ['none'], 'sp':[0.2], 's': [0,1,2,3,4], 'fr': [.2, .4, .6, .8, 1.]},
    }

    return estimators, parameters_all

def get_config_ADNI_reg():

    estimators = [get_estimators()[e] for e in ['ridge','xgbr', 'mlpr', 'PSM_ols', 'SPSM_ols']]

    # dictionary with all parameter options for all linear estimators
    parameters_all = {
        'ridge': {'alpha': [0, 0.1, 1, 5, 10], 'i': ['mean', 'iterate', 'zero'], 'sp': [0.2], 's': [0, 1, 2, 3, 4], 'm': [True], 'fr': [.2, .4, .6, .8, 1.]},
        #'xgbr': {'n_estimators': [100], 'learning_rate': [1.0], 'max_depth': [5, 10, 15], 'i': ['mean','iterate', 'zero', 'none'], 'sp':[0.2], 's': [0,1,2,3,4]},
        #'mlpr': {'hidden_layer_sizes': [10,20,30], 'activation': ['relu'], 'solver': ['adam'],
                 #'alpha': [0.0001, 0.05], 'learning_rate': ['adaptive'], 'i': ['mean','iterate', 'zero'], 'sp':[0.2], 's': [0,1,2,3,4]},
        'PSM_ols': {'alpha0':[1e8], 'alphap': [0], 'tol':[1e-5], 'penalty_main': ['l2'],
                    'min_samples_pattern':[40], 'reg_pattern_intercept': [False], 'i': ['none'], 'sp':[0.2], 's': [0,1,2,3,4], 'fr': [.2, .4, .6, .8, 1.]},
        'SPSM_ols': {'alpha0':[0, 0.1, 1, 10], 'alphap': [1, 50, 10, 100, 1000], 'tol':[1e-5], 'penalty_main':['l1'],
                    'min_samples_pattern': [0, 40], 'reg_pattern_intercept': [False], 'i': ['none'], 'sp':[0.2], 's': [0,1,2,3,4], 'fr': [.2, .4, .6, .8, 1.]}
    }

    return estimators, parameters_all



def get_config_SUPPORT_cla():

    estimators = [get_estimators()[e] for e in ['lr', 'xgbc', 'mlpc', 'PSM_lr', 'SPSM_lr']]

    # dictionary with all parameter options for all logistic estimators
    parameters_all = {
    'lr': {'penalty': ['l1', 'l2'], 'fit_intercept': [True],'i': ['mean','iterate', 'zero'], 'sp':[0.2], 's': [0,1,2,3,4], 'm': [True], 'fr': [.2, .4, .6, .8, 1.], 'op': [True]},
    #'xgbc': {'n_estimators': [100], 'learning_rate': [1.0], 'max_depth': [5, 10, 15], 'i': ['mean','mice', 'zero', 'none'], 'sp':[0.2], 's': [0,1,2,3,4]},
    #'mlpc': {'hidden_layer_sizes': [10, 20,30], 'activation': ['relu'], 'solver': ['adam'],
    #         'alpha': [0.0001, 0.05], 'learning_rate': ['adaptive'], 'i': ['mean','mice', 'zero'], 'sp':[0.2], 's': [0,1,2,3,4]},
    'PSM_lr': {'alpha0':[1e8], 'alphap': [0], 'tol':[1e-5], 'penalty_main':['l2'],
                'min_samples_pattern':[20], 'reg_pattern_intercept':[False], 'i': ['none'], 'sp':[0.2], 's': [0,1,2,3,4], 'm': [True], 'op': [True], 'fr': [.2, .4, .6, .8, 1.]},
    'SPSM_lr': {'alpha0':[0, 0.1, 1, 10, 100], 'alphap': [1, 5, 10, 100, 1000, 1e8], 'tol':[1e-5], 'penalty_main':['l1'],
                'min_samples_pattern':[0, 20], 'reg_pattern_intercept':[False], 'i': ['none'], 'sp':[0.2], 's': [0,1,2,3,4], 'm': [True], 'op': [True], 'fr': [.2, .4, .6, .8, 1.]},
    }

    return estimators, parameters_all

def get_config_SUPPORT_reg():
    estimators = [get_estimators()[e] for e in ['ridge', 'xgbr', 'mlpr', 'PSM_ols', 'SPSM_ols']]

    # dictionary with all parameter options for all linear estimators
    parameters_all = {
        'ridge': {'alpha': [0, 0.1, 1, 5, 10], 'i': ['mean', 'iterate', 'zero'], 'sp': [0.2], 's': [0, 1, 2, 3, 4],
                  'm': [True], 'fr': [.2, .4, .6, .8, 1.]},
        #'xgbr': {'n_estimators': [100], 'learning_rate': [1.0], 'max_depth': [5, 10, 15], 'i': ['mean','mice', 'zero', 'none'], 'sp':[0.2], 's': [0,1,2,3,4]},
        #'mlpr': {'hidden_layer_sizes': [10, 20,30], 'activation': ['relu'], 'solver': ['adam'],
        #         'alpha': [0.0001, 0.05], 'learning_rate': ['adaptive'], 'i': ['mean','mice', 'zero'], 'sp':[0.2], 's': [0,1,2,3,4]},
        'PSM_ols': {'alpha0':[1e8], 'alphap': [0], 'tol':[1e-5], 'penalty_main': ['l2'],
                    'min_samples_pattern':[20], 'reg_pattern_intercept': [False], 'i': ['none'], 'sp':[0.2], 's': [0,1,2,3,4], 'm': [True], 'op': [True], 'fr': [.2, .4, .6, .8, 1.]},
        'SPSM_ols': {'alpha0':[0, 0.1, 1, 10, 100], 'alphap': [1, 5, 10, 100, 1000, 1e8], 'tol':[1e-5], 'penalty_main':['l1'],
                    'min_samples_pattern': [0, 20], 'reg_pattern_intercept': [False], 'i': ['none'], 'sp':[0.2], 's': [0,1,2,3,4], 'm': [True], 'op': [True], 'fr': [.2, .4, .6, .8, 1.]}
    }

    return estimators, parameters_all
