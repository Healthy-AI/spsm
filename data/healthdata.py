import numpy as np
import pandas as pd
import os

from util.preprocessing import *

class Load_Data:
    """
    Load data
        - Load data set for House classification
        - Load data set for House regression
    """

    def load_houseClassifier(base_path='', only_psm_vars=False, mnar=False, frac=1.0):
        # read csv files and create dataframe
        df = pd.read_csv(os.path.join(base_path, 'housedata.csv'))
        df = pd.DataFrame(df)

        if frac < 1.0:
            df = df.sample(frac=frac)

        # Define X and Y
        Y = df['SalePrice']
        house_features = ['MSSubClass', 'LotFrontage', 'LotArea',
                          'LotShape', 'LandContour',
                          'LandSlope', 'Neighborhood',
                          'HouseStyle', 'YearBuilt', 'YearRemodAdd',
                          'RoofStyle', 'Foundation', 'Heating',
                          'CentralAir', 'Electrical', 'KitchenAbvGr',
                          'Functional', 'Fireplaces', 'GarageType', 'GarageCars', 'PoolArea',
                          'Fence', 'MiscFeature', 'MoSold', 'YrSold', 'SaleType',
                          'SaleCondition']
        X = df[house_features]

        df['price_diff'] = df['SalePrice'] - df['SalePrice'].median()
        Y_ = np.where(df['price_diff'] > 0, 1, 0)
        Y = Y_.reshape(-1, 1).ravel()

        return X, Y

    def load_houseRegressor(base_path='', frac=1.0):
        # read csv files and create dataframe
        df = pd.read_csv(os.path.join(base_path, 'housedata.csv'))
        df = pd.DataFrame(df)

        if frac < 1.0:
            df = df.sample(frac=frac)

        # Define X and Y
        Y = df['SalePrice'].values.reshape(-1,1)
        house_features = ['MSSubClass', 'LotFrontage', 'LotArea',
                          'LotShape', 'LandContour',
                          'LandSlope', 'Neighborhood',
                          'HouseStyle', 'YearBuilt', 'YearRemodAdd',
                          'RoofStyle', 'Foundation', 'Heating',
                          'CentralAir', 'Electrical', 'KitchenAbvGr',
                          'Functional', 'Fireplaces', 'GarageType', 'GarageCars', 'PoolArea',
                          'Fence', 'MiscFeature', 'MoSold', 'YrSold', 'SaleType',
                          'SaleCondition']
        X = df[house_features]

        # Standardize Y
        Y = StandardScaler().fit_transform(Y)

        return X, Y
