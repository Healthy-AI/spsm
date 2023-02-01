from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.linear_model import LinearRegression, Ridge
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor

from models import sharing_models
import sys

#datasets = ['ADNI_cla','ADNI_reg', 'SUPPORT_cla', 'SUPPORT_reg']
datasets = ['house_cla', 'house_reg']

def get_estimators():
    estimators = {
        'lr': LogisticRegression,
        'xgbc': XGBClassifier,
        'mlpc': MLPClassifier,
        'SM_lr': sharing_models.SharingLogisticSubModel,
        'PSM_lr': sharing_models.SharingLogisticSubModel,
        'SPSM_lr': sharing_models.SharingLogisticSubModel,
        'ridge': Ridge,
        'ols': LinearRegression,
        'xgbr': XGBRegressor,
        'mlpr': MLPRegressor,
        'SM_ols': sharing_models.SharingLinearSubModel,
        'PSM_ols': sharing_models.SharingLinearSubModel,
        'SPSM_ols': sharing_models.SharingLinearSubModel
    }
    return estimators


def get_config_ADNI_cla():

    estimators= [get_estimators()[e] for e in ['lr', 'xgbc', 'mlpc','SM_lr', 'PSM_lr', 'SPSM_lr']]

    # dictionary with all parameter options for all logistic estimators
    parameters_all = {
    'lr': {'penalty': ['l1', 'l2'],'i': ['mean', 'iterate', 'zero'], 'sp':[0.2], 's': [0,1,2,3,4]},
    'xgbc': {'n_estimators': [20, 40, 60, 80, 100], 'learning_rate': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0], 'max_depth': [2, 4, 5, 10, 15], 'lambda': [1.5], 'alpha':[0, 0.2, 0.5, 0.7],'i': ['mean','iterate', 'zero', 'none'], 'sp':[0.2], 's': [0,1,2,3,4]},
     'mlpc': {'hidden_layer_sizes': [10,20,30], 'activation': ['relu'], 'solver': ['adam' 'lbfgs'],
              'alpha': [0.0001, 0.05], 'learning_rate': ['adaptive', 'logistic'],'i': ['mean','iterate', 'zero'], 'sp':[0.2], 's': [0,1,2,3,4]},
     'PSM_lr': {'alpha0':[1e8], 'alphap': [0], 'tol':[1e-5], 'penalty_main':['l2'],
                 'min_samples_pattern':[10, 20, 40], 'reg_pattern_intercept':[False], 'i': ['none'], 'sp':[0.2], 's': [0,1,2,3,4]},
     'SPSM_lr': {'alpha0':[0, 0.1, 1, 10], 'alphap': [1, 5, 10, 12, 13, 15, 17, 20, 50, 100], 'tol':[1e-5], 'penalty_main':['l1'],
                 'min_samples_pattern':[0, 40], 'reg_pattern_intercept':[False], 'i': ['none'], 'sp':[0.2], 's': [0,1,2,3,4]},
    }

    return estimators, parameters_all

def get_config_ADNI_reg():

    estimators = [get_estimators()[e] for e in ['ridge','xgbr', 'mlpr', 'PSM_ols', 'SPSM_ols']]

    # dictionary with all parameter options for all linear estimators
    parameters_all = {
        'ridge': {'alpha': [0.7, 0.8, 0.9, 1.0],'fit_intercept': [True], 'i': ['mean','iterate', 'zero'], 'sp':[0.2], 's': [0,1,2,3,4]},
        'xgbr': {'n_estimators': [20, 40, 60, 80, 100], 'learning_rate': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
                 'max_depth': [2, 4, 5, 10, 15], 'lambda': [1.5], 'alpha': [0, 0.2, 0.5, 0.7],
                 'i': ['mean', 'iterate', 'zero', 'none'], 'sp': [0.2], 's': [0, 1, 2, 3, 4]},
        'mlpr': {'hidden_layer_sizes': [10,20,30], 'activation': ['relu'], 'solver': ['adam', 'lbfgs'],
                 'alpha': [0.0001, 0.05], 'learning_rate': ['adaptive'], 'i': ['mean','iterate', 'zero'], 'sp':[0.2], 's': [0,1,2,3,4]},
        'PSM_ols': {'alpha0':[1e8], 'alphap': [0], 'tol':[1e-5], 'penalty_main': ['l2'],
                    'min_samples_pattern':[10, 20, 40], 'reg_pattern_intercept': [False], 'i': ['none'], 'sp':[0.2], 's': [0,1,2,3,4]},
        'SPSM_ols': {'alpha0':[0, 0.1, 1, 10], 'alphap': [1, 5, 10, 12, 13, 15, 17, 20, 50, 100], 'tol':[1e-5], 'penalty_main':['l1'],
                    'min_samples_pattern': [0, 40], 'reg_pattern_intercept': [False], 'i': ['none'], 'sp':[0.2], 's': [0,1,2,3,4]}
    }

    return estimators, parameters_all

def get_config_SUPPORT_cla():

    estimators = [get_estimators()[e] for e in ['lr', 'xgbc', 'mlpc', 'PSM_lr', 'SPSM_lr']]

    # dictionary with all parameter options for all logistic estimators
    parameters_all = {
    'lr': {'penalty': ['l1', 'l2'], 'i': ['mean','iterate', 'zero'], 'sp':[0.2], 's': [0,1,2,3,4], 'm': [False], 'op': [True]},
        'xgbc': {'n_estimators': [20, 40, 60, 80, 100], 'learning_rate': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
                 'max_depth': [2, 4, 5, 10, 15], 'lambda': [1.5], 'alpha': [0, 0.2, 0.5, 0.7],
                 'i': ['mean', 'iterate', 'zero', 'none'], 'sp': [0.2], 's': [0, 1, 2, 3, 4], 'm': [True], 'op': [True]},
        'mlpc': {'hidden_layer_sizes': [10, 20,30], 'activation': ['relu', 'logistic'], 'solver': ['adam', 'lbfgs'],
              'alpha': [0.0001, 0.05], 'learning_rate': ['adaptive'], 'i': ['mean','iterate', 'zero'], 'sp':[0.2], 's': [0,1,2,3,4], 'm': [True], 'op': [True]},
     'PSM_lr': {'alpha0':[1e8], 'alphap': [0], 'tol':[1e-5], 'penalty_main':['l2'],
                 'min_samples_pattern':[20], 'reg_pattern_intercept':[False], 'i': ['none'], 'sp':[0.2], 's': [0,1,2,3,4], 'm': [True], 'op': [True]},
     'SPSM_lr': {'alpha0':[0, 0.1, 1, 5, 10, 100], 'alphap': [1, 5, 10,15, 20, 25, 30, 40,  100, 1000, 1e8], 'tol':[1e-5], 'penalty_main':['l1'],
                 'min_samples_pattern':[0, 20], 'reg_pattern_intercept':[False], 'i': ['none'], 'sp':[0.2], 's': [0,1,2,3,4], 'm': [True], 'op': [True]}
    }

    return estimators, parameters_all

def get_config_SUPPORT_reg():
    estimators = [get_estimators()[e] for e in ['ridge', 'xgbr', 'mlpr', 'PSM_ols', 'SPSM_ols']]

    # dictionary with all parameter options for all linear estimators
    parameters_all = {
        'ridge': {'alpha': [0.7, 0.8, 0.9, 1.0],'fit_intercept': [True], 'i': ['mean','iterate', 'zero'], 'sp':[0.2], 's': [0,1], 'm': [True], 'op': [True]},
        'xgbr': {'n_estimators': [20, 40], 'learning_rate': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
                 'max_depth': [2, 4, 15], 'lambda': [1.5], 'alpha': [0, 0.2, 0.5, 0.7],
                 'i': ['mean', 'iterate', 'zero', 'none'], 'sp': [0.2], 's': [0, 1, 2, 3, 4], 'm': [True], 'op': [True]},
        'mlpr': {'hidden_layer_sizes': [10, 20,30], 'activation': ['relu'], 'solver': ['adam', 'lbfgs'],'alpha': [0.0001, 0.05], 'learning_rate': ['adaptive'], 'i': ['mean','iterate', 'zero'], 'sp':[0.2], 's': [0,1,2,3,4], 'm': [True], 'op': [True]},
        'PSM_ols': {'alpha0':[1e8], 'alphap': [0], 'tol':[1e-5], 'penalty_main': ['l2'], 'min_samples_pattern':[20], 'reg_pattern_intercept': [False], 'i': ['none'], 'sp':[0.2], 's': [0,1,2,3,4], 'm': [True], 'op': [True]},
        'SPSM_ols': {'alpha0':[0, 0.1, 1,5, 10, 100], 'alphap': [1, 5, 10,15, 20, 25, 30, 40,  100, 1000, 1e8], 'tol':[1e-5], 'penalty_main':['l1'], 'min_samples_pattern': [0, 20], 'reg_pattern_intercept': [False], 'i': ['none'], 'sp':[0.2], 's': [0,1,2,3,4], 'm': [True], 'op': [True]}
    }

    return estimators, parameters_all

def get_config_house_cla():

    estimators = [get_estimators()[e] for e in ['lr', 'xgbc', 'mlpc', 'PSM_lr', 'SPSM_lr']]

    # dictionary with all parameter options for all logistic estimators
    parameters_all = {
        'lr': {'penalty': ['l1', 'l2'], 'i': ['mean','iterate', 'zero'], 'sp':[0.2], 's': [0,1], 'm': [False], 'op': [True]},
        'xgbc': {'n_estimators': [20, 40], 'learning_rate': [0.2, 0.6, 1.0],
                 'max_depth': [2, 4, 10], 'lambda': [1.5], 'alpha': [0, 0.3, 0.7],
                 'i': ['mean', 'iterate', 'zero', 'none'], 'sp': [0.2], 's': [0, 1], 'm': [True], 'op': [True]},
        'mlpc': {'hidden_layer_sizes': [10], 'activation': ['relu', 'logistic'], 'solver': ['adam', 'lbfgs'],
              'alpha': [0.0001, 0.05], 'learning_rate': ['adaptive'], 'i': ['mean','iterate', 'zero'], 'sp':[0.2], 's': [0,1], 'm': [True], 'op': [True]},
        'PSM_lr': {'alpha0':[1e8], 'alphap': [0], 'tol':[1e-5], 'penalty_main':['l2'],
                 'min_samples_pattern':[20], 'reg_pattern_intercept':[False], 'i': ['none'], 'sp':[0.2], 's': [0,1], 'm': [True], 'op': [True]},
        'SPSM_lr': {'alpha0':[0, 0.1, 1, 10], 'alphap': [10,  100, 1000, 1e8], 'tol':[1e-5], 'penalty_main':['l1'],
                 'min_samples_pattern':[0], 'reg_pattern_intercept':[False], 'i': ['none'], 'sp':[0.2], 's': [0,1], 'm': [True], 'op': [True]}
    }

    return estimators, parameters_all

def get_config_house_reg():
    estimators = [get_estimators()[e] for e in ['ridge', 'xgbr', 'mlpr', 'PSM_ols', 'SPSM_ols']]

    # dictionary with all parameter options for all linear estimators
    parameters_all = {
        'ridge': {'alpha': [0.7, 0.9],'fit_intercept': [True], 'i': ['mean','iterate', 'zero'], 'sp':[0.2], 's': [0,1], 'm': [True], 'op': [True]},
        'xgbr': {'n_estimators': [20, 40], 'learning_rate': [0.2, 0.6, 1.0],
                 'max_depth': [2, 4, 10], 'lambda': [1.5], 'alpha': [0, 0.3, 0.7],
                 'i': ['mean', 'iterate', 'zero', 'none'], 'sp': [0.2], 's': [0, 1], 'm': [True], 'op': [True]},
        'mlpr': {'hidden_layer_sizes': [10], 'activation': ['relu'], 'solver': ['adam', 'lbfgs'],'alpha': [0.0001, 0.05], 'learning_rate': ['adaptive'], 'i': ['mean','iterate', 'zero'], 'sp':[0.2], 's': [0,1], 'm': [True], 'op': [True]},
        'PSM_ols': {'alpha0':[1e8], 'alphap': [0], 'tol':[1e-5], 'penalty_main': ['l2'], 'min_samples_pattern':[20], 'reg_pattern_intercept': [False], 'i': ['none'], 'sp':[0.2], 's': [0,1], 'm': [True], 'op': [True]},
        'SPSM_ols': {'alpha0':[0, 0.1, 1, 10], 'alphap': [10, 100, 1000, 1e8], 'tol':[1e-5], 'penalty_main':['l1'], 'min_samples_pattern': [0], 'reg_pattern_intercept': [False], 'i': ['none'], 'sp':[0.2], 's': [0,1], 'm': [True], 'op': [True]}
    }

    return estimators, parameters_all