from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.linear_model import LinearRegression, Ridge
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor

from models import sharing_models
import sys

datasets = ['SYNTH_A', 'SYNTH_B']

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


def get_config_SYNTH():

    estimators = [get_estimators()[e] for e in ['ridge', 'xgbr', 'mlpr', 'PSM_ols', 'SPSM_ols']]

    # dictionary with all parameter options for all logistic estimators
    parameters_all = {
    'ridge': {'alpha': [0, 0.1, 1, 5, 10], 'i': ['mean','mice', 'zero'], 'sp':[0.2], 's': [0,1,2,3,4], 'm': [True], 'fr': [.1, .2, .4, .6, .8, 1.]},
    'xgbr': {'n_estimators': [100], 'learning_rate': [1.0], 'max_depth': [5, 10, 15], 'i': ['mean','mice', 'zero', 'none'], 'sp':[0.2], 's': [0,1,2,3,4], 'fr': [.1, .2, .4, .6, .8, 1.]},
    'mlpr': {'hidden_layer_sizes': [10, 20,30], 'activation': ['relu'], 'solver': ['adam'],
             'alpha': [0.0001, 0.05], 'learning_rate': ['adaptive'], 'i': ['mean','mice', 'zero'], 'sp':[0.2], 's': [0,1,2,3,4], 'fr': [.1, .2, .4, .6, .8, 1.]},
    'PSM_ols': {'alpha0':[1e8], 'alphap': [0, 1., 5., 10.], 'tol':[1e-5], 'penalty_main':['l2'],
                'min_samples_pattern':[20], 'reg_pattern_intercept':[False], 'i': ['none'], 'sp':[0.2], 's': [0,1,2,3,4], 'fr': [.1, .2, .4, .6, .8, 1.]},
    'SPSM_ols': {'alpha0':[0, 0.1, 1, 10, 100], 'alphap': [1, 5, 10, 100, 1e8], 'tol':[1e-5], 'penalty_main':['l1', 'l2'],
                'min_samples_pattern':[0, 20], 'reg_pattern_intercept':[False], 'i': ['none'], 'sp':[0.2], 's': [0,1,2,3,4], 'fr': [.1, .2, .4, .6, .8, 1.]},
    }

    return estimators, parameters_all
