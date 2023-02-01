import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score

from util.preprocessing import *

class Submodel:
    #Implements the Submodel
    '''def __init__(self, param1, < arguments, e.

        g., hyperparameters >):
    """
    Initializes a FancyClassifier with parameters

    Args:
        param1 (str): The first hyperparameter
    """

    # Store arguments
    self.param1 = param1'''

    def identify_patterns(self, X_bool):
        P = X_bool.drop_duplicates(subset=None, keep='first', inplace=False)
        P = P.reset_index(drop=True)
        return P

    def select_patients_in_pattern(self, X_train, X_bool_cols, P_bool_cols, p, y_train):
        pattern_match = [np.nan] * X_bool_cols.shape[0]
        for i in range(X_bool_cols.shape[0]):
            match_X_P = np.all([X_bool_cols.iloc[i, :].eq(P_bool_cols)], axis=None)
            if match_X_P == True:
                pattern_match[i] = p
        X_train['pattern_no'] = pattern_match
        y_train_p = y_train[X_train['pattern_no'] == p]
        X_pattern_p = X_train[X_train['pattern_no'] == p]
        return X_pattern_p, y_train_p

    def select_patients_cover_pattern(self, X_train, X_bool_cols, P_bool_cols, p, y_train):
        pattern_nos_extended = [np.nan] * X_bool_cols.shape[0]
        for i in range(X_bool_cols.shape[0]):
            extent_X_P = np.all(X_bool_cols.iloc[i, :] >= (P_bool_cols))
            if extent_X_P == True:
                pattern_nos_extended[i] = p
        X_train['pattern_no_extended'] = pattern_nos_extended
        y_train_p = y_train[X_train['pattern_no_extended'] == p]
        X_ext_p = X_train[X_train['pattern_no_extended'] == p]
        return X_ext_p, y_train_p

    def own_fit(self, X_train, X_bool, y_train):
        #Fits the Submodel to data X and y_train

        P = self.identify_patterns(X_bool)  # Dataframe of the missingness mask per pattern [x]


        # columns of interest of full training dataframe =predictors
        X_bool_cols = X_bool[
            ['APOE4', 'FDG', 'ABETA_bl', 'TAU_bl', 'PTAU_bl', 'ADAS13', 'LDELTOTAL', 'MMSE', 'Hippocampus_bl',
             'WholeBrain_bl', 'Entorhinal', 'Fusiform', 'ICV']]
        X_bool_cols = X_bool_cols.reset_index(drop=True)
        # columns of interest of missingness masks columns dataframe
        P_bool_cols = P[['APOE4', 'FDG', 'ABETA_bl', 'TAU_bl', 'PTAU_bl', 'ADAS13', 'LDELTOTAL', 'MMSE', 'Hippocampus_bl',
                    'WholeBrain_bl', 'Entorhinal', 'Fusiform', 'ICV']] #whole dataframe in boleean with only relevant columns without dummies

        d = len(X_bool_cols.columns)  # dimension of X(number of covariates) boolean
        nmin = 2 * d  # minimum of samples in X[p] to be submodel

        m = P_bool_cols.shape[0]  # number of patterns in P # e.g. 21

        models = []
        complete_case = [0] * m #what is this again?
        nps = []  # number of patients per pattern

        self.p_cols = []
        for p in range(m):
            print("### PATTERN ", p)
            y_train_copy = y_train.copy()
            Xp_exact, y_train_exact = self.select_patients_in_pattern(X_train, X_bool_cols, P_bool_cols.iloc[[p]], p,
                                                                      y_train_copy)  # y_train is overritten with method output
             # Dataframe of patients who match p exactly on the columns in P/X
            np_1 = Xp_exact.shape[0]  # number of patients in Xp_exact
            if np_1 < nmin:
                Xp, y_train_exact = self.select_patients_cover_pattern(X_train, X_bool_cols, P_bool_cols.iloc[[p]], p,
                                                                       y_train_copy)
                # Xp = select_patients_cover_pattern(X, P[p]) # Dataframe of patients who match p exactly or have all variables in P and more
                complete_case[p] = True
            else:
                Xp = Xp_exact
                complete_case[p] = False
            print(Xp)
            model_clf = LogisticRegression(random_state=0)

            # column selection for pattern p
            p_cols = []

            for c in P_bool_cols.columns:
                if P_bool_cols.at[p,c] == True:
                    p_cols.append(c)

            model_clf.fit(Xp[p_cols], y_train_exact) #Xp should only include columns that are measured by pattern p
            nps.append(np_1)
            models.append(model_clf)

            # Store patterns and models
            self.P = P
            self.P_bool_cols = P_bool_cols
            self.models = models #fitted models for each pattern
            self.nps = nps
            self.p_cols.append(p_cols)

        print("models saved")
        return models


    #Prediction
    def predict(self, X_test): #X_test for prediction
        #Predicts the label(s) of x

        m=self.P.shape[0] #number of patterns e.g. 21
        y_predict=[] #List for y_test (the predictions)

        for i in range(X_test.shape[0]): #go through each patient rows
            xi =X_test.iloc[i:(i+1)] #go over each row (patients)
            pi = self.find_closest_pattern(xi.iloc[0], self.P) # Finds the pattern which has missingness closest to that of xi (p has to match or be subpattern of xi's missingness).

            yi = self.models[pi].predict(xi[self.p_cols[pi]]) # E.g. pi_cols = ['Age', 'MMSE'], so you get xi[['Age', 'MMSE']] #
            y_predict.append(yi)


        return np.array(y_predict)

    def find_closest_pattern(self, xi, P):
        diff = np.zeros(P.shape[0]) #empty list of length P.shape
        for i in range(P.shape[0]): #iterate over rows
            for j in range(xi.shape[0]): #iterate over columns
                xbool = not pd.isna(xi.iloc[j])
                if (not xbool) and P.iloc[i,j]: #if xi is not true (=false) but P is true
                    diff[i]=+10000 #add very high number
                elif xbool and (not P.iloc[i,j]): #if xi is true and P is not ture
                    diff[i]+=1 #only add 1

        Ps = np.argsort(diff) #sorted by the positions in the array (=index from dataframe)
        Ps_pat = Ps[0]
        return Ps_pat

    #Evaluation
    def evaluate(self, y_predict, y_test):
        accuracy = balanced_accuracy_score(y_test, y_predict)
        f1_s = f1_score(y_test, y_predict, average='weighted')
        print('accuracy:', accuracy)
        print('f1:', f1_s)

        #save metric results in pd for each xi
        metrics = pd.DataFrame(np.array([[accuracy, f1_s]]))
        return metrics

