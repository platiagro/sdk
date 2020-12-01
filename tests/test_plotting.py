from unittest import TestCase
from uuid import uuid4

import numpy as np
import pandas as pd

from category_encoders.ordinal import OrdinalEncoder
from category_encoders.one_hot import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression


from platiagro import plot_roc_curve
from platiagro import plot_regression_error
from platiagro import plot_regression_data
from platiagro import plot_clustering_data
from platiagro import plot_data_table

 
RUN_ID = str(uuid4())

class TestPlotting(TestCase):

    def setUp(self):
        pass

    def test_roc(self):
        y_test = np.array([0, 0, 0, 2, 0, 0, 2, 1])
        
        y_prob = np.array([[8.78021024e-01, 1.21933551e-01, 4.54251082e-05],
                        [9.17431865e-01, 8.24983167e-02, 6.98183277e-05],
                        [8.56739871e-01, 1.43160592e-01, 9.95368771e-05],
                        [3.05500203e-03, 5.88313236e-01, 4.08631762e-01],
                        [7.93978912e-01, 2.05917495e-01, 1.03592501e-04],
                        [8.06967813e-01, 1.92753804e-01, 2.78382289e-04],
                        [5.79320886e-04, 2.49958000e-01, 7.49462679e-01],
                        [2.65674563e-01, 6.64852116e-01, 6.94733211e-02]])

        labels = np.array(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])


        plot_roc_curve(y_test, y_prob, labels)

    def test_regression_error(self):
        y_reg = np.array([18.2, 18.5, 23.3, 13.9, 22.8,
                               21.7, 24.8, 14.5, 13.5, 50. , 
                               18.7, 18.4, 20.2, 22.8, 24.3,
                               27.5, 18.6, 42.3, 20.5, 24.8,
                               31.7, 24.3,32.9, 14.6, 19.9,
                               50. ,  8.5, 29.4, 22.2, 18.5,
                               26.4, 50. , 24.5, 22. , 28.6,
                               31.6, 29.1, 41.3, 21.5, 31.5, 
                               35.4, 23.7, 21. , 20.1, 15.7,
                               11.9, 23.1, 20.6, 15.6, 5.6])
        
        y_reg_pred = np.array([30.00384338, 25.02556238, 30.56759672, 28.60703649, 27.94352423,
                                    25.25628446, 23.00180827, 19.53598843, 11.52363685, 18.92026211,
                                    18.99949651, 21.58679568, 20.90652153, 19.55290281, 19.28348205,
                                    19.29748321, 20.52750979, 16.91140135, 16.17801106, 18.40613603,
                                    12.52385753, 17.67103669, 15.83288129, 13.80628535, 15.67833832,
                                    13.38668561, 15.46397655, 14.70847428, 19.54737285, 20.8764282 ,
                                    11.45511759, 18.05923295,  8.81105736, 14.28275814, 13.70675891,
                                    23.81463526, 22.34193708, 23.10891142, 22.91502612, 31.35762569,
                                    34.21510225, 28.02056414, 25.20386628, 24.60979273, 22.94149176,
                                    22.09669817, 20.42320032, 18.03655088,  9.10655377, 17.20607751])
        plot_regression_error(y_reg ,y_reg_pred)
        
    
    def test_regression_data(self):
        x_train = np.array([[1.58760e-01, 0.00000e+00, 1.08100e+01, 0.00000e+00, 4.13000e-01,
                            5.96100e+00, 1.75000e+01, 5.28730e+00, 4.00000e+00, 3.05000e+02,
                            1.92000e+01, 3.76940e+02, 9.88000e+00],
                        [6.80117e+00, 0.00000e+00, 1.81000e+01, 0.00000e+00, 7.13000e-01,
                            6.08100e+00, 8.44000e+01, 2.71750e+00, 2.40000e+01, 6.66000e+02,
                            2.02000e+01, 3.96900e+02, 1.47000e+01],
                        [7.16500e-02, 0.00000e+00, 2.56500e+01, 0.00000e+00, 5.81000e-01,
                            6.00400e+00, 8.41000e+01, 2.19740e+00, 2.00000e+00, 1.88000e+02,
                            1.91000e+01, 3.77670e+02, 1.42700e+01],
                        [3.65900e-02, 2.50000e+01, 4.86000e+00, 0.00000e+00, 4.26000e-01,
                            6.30200e+00, 3.22000e+01, 5.40070e+00, 4.00000e+00, 2.81000e+02,
                            1.90000e+01, 3.96900e+02, 6.72000e+00],
                        [6.41700e-02, 0.00000e+00, 5.96000e+00, 0.00000e+00, 4.99000e-01,
                            5.93300e+00, 6.82000e+01, 3.36030e+00, 5.00000e+00, 2.79000e+02,
                            1.92000e+01, 3.96900e+02, 9.68000e+00],
                        [9.29900e-02, 0.00000e+00, 2.56500e+01, 0.00000e+00, 5.81000e-01,
                            5.96100e+00, 9.29000e+01, 2.08690e+00, 2.00000e+00, 1.88000e+02,
                            1.91000e+01, 3.78090e+02, 1.79300e+01],
                        [1.28023e+01, 0.00000e+00, 1.81000e+01, 0.00000e+00, 7.40000e-01,
                            5.85400e+00, 9.66000e+01, 1.89560e+00, 2.40000e+01, 6.66000e+02,
                            2.02000e+01, 2.40520e+02, 2.37900e+01],
                        [3.75780e-01, 0.00000e+00, 1.05900e+01, 1.00000e+00, 4.89000e-01,
                            5.40400e+00, 8.86000e+01, 3.66500e+00, 4.00000e+00, 2.77000e+02,
                            1.86000e+01, 3.95240e+02, 2.39800e+01],
                        [2.00900e-02, 9.50000e+01, 2.68000e+00, 0.00000e+00, 4.16100e-01,
                            8.03400e+00, 3.19000e+01, 5.11800e+00, 4.00000e+00, 2.24000e+02,
                            1.47000e+01, 3.90550e+02, 2.88000e+00],
                        [3.58090e-01, 0.00000e+00, 6.20000e+00, 1.00000e+00, 5.07000e-01,
                            6.95100e+00, 8.85000e+01, 2.86170e+00, 8.00000e+00, 3.07000e+02,
                            1.74000e+01, 3.91700e+02, 9.71000e+00]])

        y_train = np.array([21.7, 20. , 20.3, 24.8, 18.9, 
                            20.5, 10.8, 19.3, 50. , 26.7])

        x_test = np.array([[7.2580e-01, 0.0000e+00, 8.1400e+00, 0.0000e+00, 5.3800e-01,
                            5.7270e+00, 6.9500e+01, 3.7965e+00, 4.0000e+00, 3.0700e+02,
                            2.1000e+01, 3.9095e+02, 1.1280e+01],
                        [1.4231e-01, 0.0000e+00, 1.0010e+01, 0.0000e+00, 5.4700e-01,
                            6.2540e+00, 8.4200e+01, 2.2565e+00, 6.0000e+00, 4.3200e+02,
                            1.7800e+01, 3.8874e+02, 1.0450e+01],
                        [4.5600e-02, 0.0000e+00, 1.3890e+01, 1.0000e+00, 5.5000e-01,
                            5.8880e+00, 5.6000e+01, 3.1121e+00, 5.0000e+00, 2.7600e+02,
                            1.6400e+01, 3.9280e+02, 1.3510e+01],
                        [1.5288e+01, 0.0000e+00, 1.8100e+01, 0.0000e+00, 6.7100e-01,
                            6.6490e+00, 9.3300e+01, 1.3449e+00, 2.4000e+01, 6.6600e+02,
                            2.0200e+01, 3.6302e+02, 2.3240e+01],
                        [7.6162e-01, 2.0000e+01, 3.9700e+00, 0.0000e+00, 6.4700e-01,
                            5.5600e+00, 6.2800e+01, 1.9865e+00, 5.0000e+00, 2.6400e+02,
                            1.3000e+01, 3.9240e+02, 1.0450e+01]])

        y_test  = np.array([18.2, 18.5, 23.3, 13.9, 22.8])

        y_pred = np.array([30.00384338, 25.02556238, 30.56759672, 28.60703649, 27.94352423])

        columns = np.array(['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad',
                            'tax', 'ptratio', 'black', 'lstat'],dtype=object)


        numerical_indexes =  np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12])
        non_numerical_indexes = np.array([],int)
        one_hot_indexes_after_handle_missing_values = np.array([],int)
        ordinal_indexes_after_handle_missing_values = np.array([],int)
        fit_intercept = True 

        pipeline = Pipeline(
            steps=[
                (
                    "handle_missing_values",
                    ColumnTransformer(
                        [
                            ("imputer_mean", SimpleImputer(strategy="mean"), numerical_indexes),
                            (
                                "imputer_mode",
                                SimpleImputer(strategy="most_frequent"),
                                non_numerical_indexes,
                            ),
                        ],
                        remainder="drop",
                    ),
                ),
                (
                    "handle categorical features",
                    ColumnTransformer(
                        [
                            (
                                "feature_encoder_ordinal",
                                OrdinalEncoder(),
                                ordinal_indexes_after_handle_missing_values,
                            ),
                            (
                                "feature_encoder_onehot",
                                OneHotEncoder(),
                                one_hot_indexes_after_handle_missing_values,
                            ),
                        ],
                        remainder="passthrough",
                    ),
                ),
                ("estimator", LinearRegression(fit_intercept=fit_intercept)),
            ]
        )

        pipeline.fit(x_train, y_train)
        plot_regression_data(pipeline,columns, x_train, y_train, x_test, y_test, y_pred)

    def test_clustering_data(self):


        x_test  = np.array([[5.1, 3.5, 1.4, 0.2, 'Iris-setosa'],
                            [4.9, 3.0, 1.4, 0.2, 'Iris-setosa'],
                            [4.7, 3.2, 1.3, 0.2, 'Iris-setosa'],
                            [4.6, 3.1, 1.5, 0.2, 'Iris-setosa'],
                            [5.0, 3.6, 1.4, 0.2, 'Iris-setosa'],
                            [5.4, 3.9, 1.7, 0.4, 'Iris-setosa'],
                            [4.6, 3.4, 1.4, 0.3, 'Iris-setosa'],
                            [5.0, 3.4, 1.5, 0.2, 'Iris-setosa'],
                            [4.4, 2.9, 1.4, 0.2, 'Iris-setosa'],
                            [4.9, 3.1, 1.5, 0.1, 'Iris-setosa'],
                            [5.4, 3.7, 1.5, 0.2, 'Iris-setosa'],
                            [4.8, 3.4, 1.6, 0.2, 'Iris-setosa'],
                            [4.8, 3.0, 1.4, 0.1, 'Iris-setosa'],
                            [4.3, 3.0, 1.1, 0.1, 'Iris-setosa'],
                            [5.8, 4.0, 1.2, 0.2, 'Iris-setosa'],
                            [5.7, 4.4, 1.5, 0.4, 'Iris-setosa'],
                            [5.4, 3.9, 1.3, 0.4, 'Iris-setosa'],
                            [5.1, 3.5, 1.4, 0.3, 'Iris-setosa'],
                            [5.7, 3.8, 1.7, 0.3, 'Iris-setosa'],
                            [5.1, 3.8, 1.5, 0.3, 'Iris-setosa'],
                            [5.4, 3.4, 1.7, 0.2, 'Iris-setosa'],
                            [5.1, 3.7, 1.5, 0.4, 'Iris-setosa'],
                            [4.6, 3.6, 1.0, 0.2, 'Iris-setosa'],
                            [5.1, 3.3, 1.7, 0.5, 'Iris-setosa'],
                            [4.8, 3.4, 1.9, 0.2, 'Iris-setosa'],
                            [5.0, 3.0, 1.6, 0.2, 'Iris-setosa'],
                            [5.0, 3.4, 1.6, 0.4, 'Iris-setosa'],
                            [5.2, 3.5, 1.5, 0.2, 'Iris-setosa'],
                            [5.2, 3.4, 1.4, 0.2, 'Iris-setosa'],
                            [4.7, 3.2, 1.6, 0.2, 'Iris-setosa'],
                            [4.8, 3.1, 1.6, 0.2, 'Iris-setosa'],
                            [5.4, 3.4, 1.5, 0.4, 'Iris-setosa'],
                            [5.2, 4.1, 1.5, 0.1, 'Iris-setosa'],
                            [5.5, 4.2, 1.4, 0.2, 'Iris-setosa'],
                            [4.9, 3.1, 1.5, 0.1, 'Iris-setosa'],
                            [5.0, 3.2, 1.2, 0.2, 'Iris-setosa'],
                            [5.5, 3.5, 1.3, 0.2, 'Iris-setosa'],
                            [4.9, 3.1, 1.5, 0.1, 'Iris-setosa'],
                            [4.4, 3.0, 1.3, 0.2, 'Iris-setosa'],
                            [5.1, 3.4, 1.5, 0.2, 'Iris-setosa'],
                            [5.0, 3.5, 1.3, 0.3, 'Iris-setosa'],
                            [4.5, 2.3, 1.3, 0.3, 'Iris-setosa'],
                            [4.4, 3.2, 1.3, 0.2, 'Iris-setosa'],
                            [5.0, 3.5, 1.6, 0.6, 'Iris-setosa'],
                            [5.1, 3.8, 1.9, 0.4, 'Iris-setosa'],
                            [4.8, 3.0, 1.4, 0.3, 'Iris-setosa'],
                            [5.1, 3.8, 1.6, 0.2, 'Iris-setosa'],
                            [4.6, 3.2, 1.4, 0.2, 'Iris-setosa'],
                            [5.3, 3.7, 1.5, 0.2, 'Iris-setosa'],
                            [5.0, 3.3, 1.4, 0.2, 'Iris-setosa'],
                            [7.0, 3.2, 4.7, 1.4, 'Iris-versicolor'],
                            [6.4, 3.2, 4.5, 1.5, 'Iris-versicolor'],
                            [6.9, 3.1, 4.9, 1.5, 'Iris-versicolor'],
                            [5.5, 2.3, 4.0, 1.3, 'Iris-versicolor'],
                            [6.5, 2.8, 4.6, 1.5, 'Iris-versicolor'],
                            [5.7, 2.8, 4.5, 1.3, 'Iris-versicolor'],
                            [6.3, 3.3, 4.7, 1.6, 'Iris-versicolor'],
                            [4.9, 2.4, 3.3, 1.0, 'Iris-versicolor'],
                            [6.6, 2.9, 4.6, 1.3, 'Iris-versicolor'],
                            [5.2, 2.7, 3.9, 1.4, 'Iris-versicolor'],
                            [5.0, 2.0, 3.5, 1.0, 'Iris-versicolor'],
                            [5.9, 3.0, 4.2, 1.5, 'Iris-versicolor'],
                            [6.0, 2.2, 4.0, 1.0, 'Iris-versicolor'],
                            [6.1, 2.9, 4.7, 1.4, 'Iris-versicolor'],
                            [5.6, 2.9, 3.6, 1.3, 'Iris-versicolor'],
                            [6.7, 3.1, 4.4, 1.4, 'Iris-versicolor'],
                            [5.6, 3.0, 4.5, 1.5, 'Iris-versicolor'],
                            [5.8, 2.7, 4.1, 1.0, 'Iris-versicolor'],
                            [6.2, 2.2, 4.5, 1.5, 'Iris-versicolor'],
                            [5.6, 2.5, 3.9, 1.1, 'Iris-versicolor'],
                            [5.9, 3.2, 4.8, 1.8, 'Iris-versicolor'],
                            [6.1, 2.8, 4.0, 1.3, 'Iris-versicolor'],
                            [6.3, 2.5, 4.9, 1.5, 'Iris-versicolor'],
                            [6.1, 2.8, 4.7, 1.2, 'Iris-versicolor'],
                            [6.4, 2.9, 4.3, 1.3, 'Iris-versicolor'],
                            [6.6, 3.0, 4.4, 1.4, 'Iris-versicolor'],
                            [6.8, 2.8, 4.8, 1.4, 'Iris-versicolor'],
                            [6.7, 3.0, 5.0, 1.7, 'Iris-versicolor'],
                            [6.0, 2.9, 4.5, 1.5, 'Iris-versicolor'],
                            [5.7, 2.6, 3.5, 1.0, 'Iris-versicolor'],
                            [5.5, 2.4, 3.8, 1.1, 'Iris-versicolor'],
                            [5.5, 2.4, 3.7, 1.0, 'Iris-versicolor'],
                            [5.8, 2.7, 3.9, 1.2, 'Iris-versicolor'],
                            [6.0, 2.7, 5.1, 1.6, 'Iris-versicolor'],
                            [5.4, 3.0, 4.5, 1.5, 'Iris-versicolor'],
                            [6.0, 3.4, 4.5, 1.6, 'Iris-versicolor'],
                            [6.7, 3.1, 4.7, 1.5, 'Iris-versicolor'],
                            [6.3, 2.3, 4.4, 1.3, 'Iris-versicolor'],
                            [5.6, 3.0, 4.1, 1.3, 'Iris-versicolor'],
                            [5.5, 2.5, 4.0, 1.3, 'Iris-versicolor'],
                            [5.5, 2.6, 4.4, 1.2, 'Iris-versicolor'],
                            [6.1, 3.0, 4.6, 1.4, 'Iris-versicolor'],
                            [5.8, 2.6, 4.0, 1.2, 'Iris-versicolor'],
                            [5.0, 2.3, 3.3, 1.0, 'Iris-versicolor'],
                            [5.6, 2.7, 4.2, 1.3, 'Iris-versicolor'],
                            [5.7, 3.0, 4.2, 1.2, 'Iris-versicolor'],
                            [5.7, 2.9, 4.2, 1.3, 'Iris-versicolor'],
                            [6.2, 2.9, 4.3, 1.3, 'Iris-versicolor'],
                            [5.1, 2.5, 3.0, 1.1, 'Iris-versicolor'],
                            [5.7, 2.8, 4.1, 1.3, 'Iris-versicolor'],
                            [6.3, 3.3, 6.0, 2.5, 'Iris-virginica'],
                            [5.8, 2.7, 5.1, 1.9, 'Iris-virginica'],
                            [7.1, 3.0, 5.9, 2.1, 'Iris-virginica'],
                            [6.3, 2.9, 5.6, 1.8, 'Iris-virginica'],
                            [6.5, 3.0, 5.8, 2.2, 'Iris-virginica'],
                            [7.6, 3.0, 6.6, 2.1, 'Iris-virginica'],
                            [4.9, 2.5, 4.5, 1.7, 'Iris-virginica'],
                            [7.3, 2.9, 6.3, 1.8, 'Iris-virginica'],
                            [6.7, 2.5, 5.8, 1.8, 'Iris-virginica'],
                            [7.2, 3.6, 6.1, 2.5, 'Iris-virginica'],
                            [6.5, 3.2, 5.1, 2.0, 'Iris-virginica'],
                            [6.4, 2.7, 5.3, 1.9, 'Iris-virginica'],
                            [6.8, 3.0, 5.5, 2.1, 'Iris-virginica'],
                            [5.7, 2.5, 5.0, 2.0, 'Iris-virginica'],
                            [5.8, 2.8, 5.1, 2.4, 'Iris-virginica'],
                            [6.4, 3.2, 5.3, 2.3, 'Iris-virginica'],
                            [6.5, 3.0, 5.5, 1.8, 'Iris-virginica'],
                            [7.7, 3.8, 6.7, 2.2, 'Iris-virginica'],
                            [7.7, 2.6, 6.9, 2.3, 'Iris-virginica'],
                            [6.0, 2.2, 5.0, 1.5, 'Iris-virginica'],
                            [6.9, 3.2, 5.7, 2.3, 'Iris-virginica'],
                            [5.6, 2.8, 4.9, 2.0, 'Iris-virginica'],
                            [7.7, 2.8, 6.7, 2.0, 'Iris-virginica'],
                            [6.3, 2.7, 4.9, 1.8, 'Iris-virginica'],
                            [6.7, 3.3, 5.7, 2.1, 'Iris-virginica'],
                            [7.2, 3.2, 6.0, 1.8, 'Iris-virginica'],
                            [6.2, 2.8, 4.8, 1.8, 'Iris-virginica'],
                            [6.1, 3.0, 4.9, 1.8, 'Iris-virginica'],
                            [6.4, 2.8, 5.6, 2.1, 'Iris-virginica'],
                            [7.2, 3.0, 5.8, 1.6, 'Iris-virginica'],
                            [7.4, 2.8, 6.1, 1.9, 'Iris-virginica'],
                            [7.9, 3.8, 6.4, 2.0, 'Iris-virginica'],
                            [6.4, 2.8, 5.6, 2.2, 'Iris-virginica'],
                            [6.3, 2.8, 5.1, 1.5, 'Iris-virginica'],
                            [6.1, 2.6, 5.6, 1.4, 'Iris-virginica'],
                            [7.7, 3.0, 6.1, 2.3, 'Iris-virginica'],
                            [6.3, 3.4, 5.6, 2.4, 'Iris-virginica'],
                            [6.4, 3.1, 5.5, 1.8, 'Iris-virginica'],
                            [6.0, 3.0, 4.8, 1.8, 'Iris-virginica'],
                            [6.9, 3.1, 5.4, 2.1, 'Iris-virginica'],
                            [6.7, 3.1, 5.6, 2.4, 'Iris-virginica'],
                            [6.9, 3.1, 5.1, 2.3, 'Iris-virginica'],
                            [5.8, 2.7, 5.1, 1.9, 'Iris-virginica'],
                            [6.8, 3.2, 5.9, 2.3, 'Iris-virginica'],
                            [6.7, 3.3, 5.7, 2.5, 'Iris-virginica'],
                            [6.7, 3.0, 5.2, 2.3, 'Iris-virginica'],
                            [6.3, 2.5, 5.0, 1.9, 'Iris-virginica'],
                            [6.5, 3.0, 5.2, 2.0, 'Iris-virginica'],
                            [6.2, 3.4, 5.4, 2.3, 'Iris-virginica'],
                            [5.9, 3.0, 5.1, 1.8, 'Iris-virginica']])
        
        df = pd.DataFrame(x_test,columns = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species'])

        y_pred = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


        numerical_indexes  = np.array([0, 1, 2, 3])
        non_numerical_indexes = np.array([4])
        non_numerical_indexes_after_handle_missing_values = np.array([4])
        n_clusters = 3
        n_init = 10
        max_iter = 300


        pipeline = Pipeline(
            steps=[
                (
                    "handle_missing_values",
                    ColumnTransformer(
                        [
                            ("imputer_mean", SimpleImputer(strategy="mean"), numerical_indexes),
                            (
                                "imputer_mode",
                                SimpleImputer(strategy="most_frequent"),
                                non_numerical_indexes,
                            ),
                        ],
                        remainder="drop",
                    ),
                ),
                (
                    "handle_categorical_features",
                    ColumnTransformer(
                        [
                            (
                                "feature_encoder",
                                OrdinalEncoder(),
                                non_numerical_indexes_after_handle_missing_values,
                            )
                        ],
                        remainder="passthrough",
                    ),
                ),
                (
                    "estimator",
                    KMeans(
                        n_clusters=n_clusters,
                        n_init=n_init,
                        max_iter=max_iter
                    ),
                ),
            ]
        )

        _ = pipeline.fit_transform(df)
       

        plot_clustering_data(pipeline, df, y_pred)

    def test_data_table(self):
        data = {'col_1': [3, 2, 1, 0], 'col_2': ['a', 'b', 'c', 'd']}
        df = pd.DataFrame.from_dict(data)
        plot_data_table( df , col_width=3.0, row_height=0.625, font_size=8, 
                header_color="#40466e", row_colors=["#f1f1f2", "w"], edge_color="w", bbox=[0, 0, 1, 1],
                header_columns=0, column_quantity=40)
