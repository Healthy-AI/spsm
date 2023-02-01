import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer

from sklearn.preprocessing._encoders import _BaseEncoder
from sklearn.preprocessing import OneHotEncoder

#set continous columns

c_cont_house = ['MSSubClass', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'KitchenAbvGr', 'Fireplaces', 'GarageCars', 'PoolArea', 'MoSold', 'YrSold']


class Standardizer(StandardScaler):
    """
    Standardizes a subset of columns using the scikit-learn StandardScaler
    """

    def __init__(self, copy=True, with_mean=True, with_std=True, columns=None, ignore_missing=True):
        StandardScaler.__init__(self, copy=copy, with_mean=with_mean, with_std=with_std)
        self.columns = columns
        self.ignore_missing = ignore_missing

    def fit(self, X, y=None):
        columns = X.columns if self.columns is None else self.columns

        StandardScaler.fit(self, X[columns], y)

        return self

    def transform(self, X, copy=None):
        columns = X.columns if self.columns is None else self.columns

        Xn = X.copy()
        if self.ignore_missing:
            columns_sub = [c for c in columns if c in X.columns]
            columns_mis = [c for c in columns if c not in X.columns]

            if len(columns_sub) == 0:
                return X

            Xt = X.copy()
            Xt[columns_mis] = 0
            try:
                Xt = StandardScaler.transform(self, Xt[columns_sub + columns_mis], copy=copy)
            except:
                print(columns_sub + columns_mis)
                print(Xt[columns_sub + columns_mis])

            Xt = Xt[:, :len(columns_sub)]
            Xn.loc[:, columns_sub] = Xt
        else:
            print('here are columns',columns)
            Xt = StandardScaler.transform(self, X[columns], copy=copy)
            Xn.loc[:, columns] = Xt

        return Xn

    def inverse_transform(self, X, copy=None):
        columns = X.columns if self.columns is None else self.columns

        if self.ignore_missing:
            columns_sub = [c for c in columns if c in X.columns]
            Xn = self.inverse_transform_single(X, columns_sub, copy=copy)
        else:
            Xt = StandardScaler.inverse_transform(self, X[columns], copy=copy)
            Xn = X.copy()
            Xn.loc[:, columns] = Xt

        return Xn

    def inverse_transform_single(self, Xs, columns, copy=None):
        X = pd.DataFrame(np.zeros((Xs.shape[0], len(self.columns))), columns=self.columns)
        X[columns] = Xs[columns]

        Xt = StandardScaler.inverse_transform(self, X[self.columns], copy=copy)
        X.loc[:, self.columns] = Xt

        return X[columns]

    def fit_transform(self, X, y=None, **fit_params):
        Xt = StandardScaler.fit_transform(self, X, y, **fit_params)
        return Xt


class OneHotEncoderMissing(_BaseEncoder):

    def __init__(self, *,
                 keep_nan=True,
                 return_df=True,
                 sparse=False,
                 categories="auto",
                 drop=None,
                 dtype=np.float64,
                 handle_unknown="error"):

        self.ohe = OneHotEncoder(sparse=sparse, categories=categories, drop=drop,
                                 dtype=dtype, handle_unknown=handle_unknown)

        if sparse:
            raise Exception('Sparse mode not supported.')

        self.keep_nan = keep_nan
        self.return_df = return_df
        self.sparse = sparse
        self.categories = categories
        self.drop = drop
        self.dtype = dtype
        self.handle_unknown = handle_unknown

    def fit(self, X, y=None):
        self.ohe.fit(X)
        return self

    def fit_transform(self, X, y=None):
        self.ohe.fit(X)
        return self.transform(X)

    def get_feature_names_out(self, input_features=None):
        if self.keep_nan:
            return [c for c in self.ohe.get_feature_names_out(input_features) if not c[-4:] == '_nan']
        else:
            return self.ohe.get_feature_names_out(input_features)

    def inverse_transform(self, X):
        cols = self.ohe.feature_names_in_

        if self.keep_nan:
            X_t = X.copy()
            if not isinstance(X_t, pd.DataFrame):
                X_t = pd.DataFrame(X_t, columns=self.get_feature_names_out())

            oh_cols = self.ohe.get_feature_names_out()

            for c in cols:
                c_nan = '%s_nan' % c
                if c_nan in oh_cols:
                    cs_notnan = [ohc for ohc in oh_cols if (c + '_') in ohc and '_nan' not in c]
                    X_t[c_nan] = 1 * (X_t[cs_notnan[0]].isna())
                    X_t[cs_notnan] = X_t[cs_notnan].fillna(0)

            X = X_t[oh_cols]

        X_i = self.ohe.inverse_transform(X)
        if self.return_df:
            if isinstance(X, pd.DataFrame):
                return pd.DataFrame(X_i, columns=cols, index=X.index)
            else:
                return pd.DataFrame(X_i, columns=cols)
            return
        else:
            return X_i

    def transform(self, X):
        X_t = self.ohe.transform(X)

        oh_cols = self.ohe.get_feature_names_out()

        if not self.keep_nan:
            if self.return_df:
                if isinstance(X, pd.DataFrame):
                    return pd.DataFrame(X_t, columns=oh_cols, index=X.index)
                else:
                    return pd.DataFrame(X_t, columns=oh_cols)
            else:
                return X_t

        if isinstance(X, pd.DataFrame):
            df_t = pd.DataFrame(X_t, columns=oh_cols, index=X.index)
        else:
            df_t = pd.DataFrame(X_t, columns=oh_cols)

        cols = self.ohe.feature_names_in_

        for c in cols:
            c_nan = '%s_nan' % c
            if c_nan in oh_cols:
                cs_nan = [ohc for ohc in oh_cols if (c + '_') in ohc]
                df_t.loc[df_t[c_nan] > 0, cs_nan] = np.nan
                df_t.drop(columns=[c_nan], inplace=True)

        if self.return_df:
            return df_t
        else:
            return df_t.values


class Preprocess:
    def __init__(self):
        self.columns = None


    def encoding_house(self, X_train, classification, frame, encoder=None):
        # categorical columns
        X_cate = X_train[['LotShape', 'LandContour', 'LandSlope', 'Neighborhood', 'HouseStyle', 'RoofStyle', 'Foundation', 'Heating',
             'CentralAir', 'Electrical', 'Functional', 'GarageType', 'Fence', 'MiscFeature', 'SaleType',
             'SaleCondition']]

        # fit onehotencoder on categorical input
        if frame == 'train':
            encoder = OneHotEncoderMissing(handle_unknown="ignore", sparse=False).fit(X_cate)

        # Get column name to be added to df since OneHotEncoder requires array
        column_names = encoder.get_feature_names_out(
            ['LotShape', 'LandContour', 'LandSlope', 'Neighborhood', 'HouseStyle', 'RoofStyle', 'Foundation', 'Heating',
             'CentralAir', 'Electrical', 'Functional', 'GarageType', 'Fence', 'MiscFeature', 'SaleType',
             'SaleCondition'])
        self.columns = column_names

        # Combine column names and encoded df
        X_ca_encoded = pd.DataFrame(encoder.transform(X_cate), columns=column_names, index=X_cate.index)

        # Select numberic columns in df
        global c_cont_house
        X_conti = X_train[c_cont_house]

        # combine onehotencoded df with continous df
        X_train = X_conti.join(X_ca_encoded)

        return X_train, encoder


    def imputer(self, X_train, imputation_string):  # "iterate,mean, zero, none"
        if imputation_string == 'mean':
            I = SimpleImputer(missing_values=np.nan, strategy="mean").fit(X_train) #mean imputation
        elif imputation_string == 'iterate':
            I = IterativeImputer(initial_strategy='mean',max_iter = 10, imputation_order='random', add_indicator=False).fit(X_train)
            #sample_posterior = True, when run with various seeds for multiple imputation
        elif imputation_string == 'zero':
            I = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=0).fit(X_train)
        elif imputation_string == 'none':
            I = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=np.nan).fit(X_train)
        else:
            raise Exception('Unknown Imputation method; %s' % (imputation_string))
        return I

    # @staticmethod

    def preprocessing_SYNTH(self, X, Y, imputation_str, split):

        # Train test split for split
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=split)

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=split)

        # Fit imputation
        I = self.imputer(X_train, imputation_str)

        #Fit Standardizer
        S = Standardizer(columns=X.columns).fit(X_train)

        return None, I, S, X_train, X_test, y_train, y_test, X_val, y_val

    def preprocessing_house(self, X, Y, imputation_str, classification, split):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=split)

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=split)

        X_train, encoder = self.encoding_house(X_train, classification, 'train')

        # Fit imputation
        I = self.imputer(X_train, imputation_str)

        c_cont = c_cont_house

        # Fit Standardizer
        S = Standardizer(columns=c_cont).fit(X_train)

        return encoder, S, I, X_train, X_test, y_train, y_test, X_val, y_val

