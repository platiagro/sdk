from unittest import TestCase
from uuid import uuid4

import numpy as np
import pandas as pd
import shap
from shap.plots._labels import labels
        

from category_encoders.ordinal import OrdinalEncoder
from category_encoders.one_hot import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from platiagro.plotting import plot_roc_curve
from platiagro.plotting import plot_regression_error
from platiagro.plotting import plot_prediction_diff
from platiagro.plotting import plot_sorted_prediction_diff
from platiagro.plotting import plot_absolute_error
from platiagro.plotting import plot_probability_error
from platiagro.plotting import plot_segment_error
from platiagro.plotting import plot_regression_data
from platiagro.plotting import plot_classification_data
from platiagro.plotting import plot_matrix
from platiagro.plotting import plot_common_metrics
from platiagro.plotting import plot_clustering_data
from platiagro.plotting import plot_data_table
from platiagro.plotting import plot_line_graphs_overlayed
from platiagro.plotting import plot_line_subgraphs_alongisde
from platiagro.plotting import plot_simple_line_graph
from platiagro.plotting import plot_simple_line_graph
from platiagro.plotting import plot_shap_classification_summary
from platiagro.plotting import plot_residues
from platiagro.plotting import plot_model_coef_weight

from .test_util import get_iris, get_boston

 
RUN_ID = str(uuid4())

iris = get_iris()
boston = get_boston()


class TestPlotting(TestCase):

    def setUp(self):
        pass


    def test_roc(self):

        labels = np.unique(iris['target'])

        plot_roc_curve(iris['target_test'],
                       iris['target_probability'],
                       labels)


    def test_regression_error(self):
        
        plot_regression_error(boston['target_test'],
                              boston['target_predicted'])


    def test_prediction_diff(self):

        plot_prediction_diff(boston['target_test'],
                             boston['target_predicted'])

    
    def test_sorted_prediction_diff(self):

        plot_sorted_prediction_diff(boston['target_test'],
                                    boston['target_predicted'])


    def test_absolute_error(self):

        plot_absolute_error(boston['target_test'],
                            boston['target_predicted'])


    def test_probability_error(self):

        plot_probability_error(boston['target_test'],
                               boston['target_predicted'])


    def test_segment_error(self):

        plot_segment_error(boston['target_test'],
                           boston['target_predicted'])
        
    
    def test_regression_data(self):

        plot_regression_data(boston['regression_pipeline'], 
                             boston['features_columns'],
                             boston['features_train'],
                             boston['target_train'],
                             boston['features_test'],
                             boston['target_test'],
                             boston['target_predicted'])


    def test_classification_data(self):

        plot_classification_data(iris['classification_pipeline'], 
                                 iris['features_columns'], 
                                 iris['features_train'], 
                                 iris['target_train'], 
                                 iris['features_test'], 
                                 iris['target_test'], 
                                 iris['target_predicted'])


    def test_matrix(self):

        # computes confusion matrix
        labels = np.unique(iris['target_encoded'])
        data = confusion_matrix(iris['target_test'],
                                iris['target_predicted'],
                                labels=labels)

        # puts matrix in pandas.DataFrame for better format
        labels_dec = np.unique(iris['target'])
        df = pd.DataFrame(data, columns=labels_dec, index=labels_dec)

        plot_matrix(df)


    def test_common_metrics(self):

        labels = np.unique(iris['target_encoded'])
        labels_dec = np.unique(iris['target'])

        plot_common_metrics(iris['target_test'],
                            iris['target_predicted'],
                            labels,
                            labels_dec)


    def test_clustering_data(self):
        
        plot_clustering_data(iris['clustering_pipeline'], 
                             iris['dataset_columns'], 
                             iris['features'],
                             iris['clusters'])


    def test_data_table(self):

        data = {'col_1': [3, 2, 1, 0], 
                'col_2': ['a', 'b', 'c', 'd']}

        df = pd.DataFrame.from_dict(data)

        plot_data_table(df)


    def test_line_subgraphs_alongisde(self):
        x1 = np.linspace(0, 10 - 2 * 1, 200) + 1
        y1 = np.sin(x1) + 1.0 + 1
        x2 = np.linspace(0, 10 - 2 * 3, 200) + 2
        y2 = np.sin(x2) + 1.0 + 2
        
        with self.assertRaises(ValueError):
            x_list = [np.array(x1)]
            y_list = [np.array(y2)]
            plot_line_subgraphs_alongisde(x_list,
                                        y_list,
                                        x_axe_names= ["x"],
                                        y_axe_names = ["y"],
                                        col_wrap=1,
                                        suptitle="Train Performance",
                                        subtitles = ['Loss'],
                                        subplot_size = (10,10))

        with self.assertRaises(ValueError):
            x_list = [np.array(x1),np.array(x2)]
            y_list = [np.array(y2)]
            plot_line_subgraphs_alongisde(x_list,
                                        y_list,
                                        x_axe_names= ["x"],
                                        y_axe_names = ["y"],
                                        col_wrap=1,
                                        suptitle="",
                                        subtitles = ['Loss'],
                                        subplot_size = (10,10))

        



        x_list = [np.array(x1),np.array(x2),np.array(x2)]
        y_list = [np.array(y1),np.array(y2),np.array(y2)]
        


        plot_line_subgraphs_alongisde(x_list,
                                    y_list,
                                    x_axe_names= ["x"],
                                    y_axe_names = ["y"],
                                    col_wrap=1,
                                    suptitle="Train Performance",
                                    subtitles = ['Loss','Acurácia','Outro'],
                                    subplot_size = (10,10))

    def test_line_graphs_overlayed(self):
        x1 = np.linspace(0, 10 - 2 * 1, 200) + 1
        y1 = np.sin(x1) + 1.0 + 1
        x2 = np.linspace(0, 10 - 2 * 3, 200) + 2
        y2 = np.sin(x2) + 1.0 + 2
        



        with self.assertRaises(ValueError):
            x_list = [np.array(x1)]
            y_list = [np.array(y1)]
            legends = ["legend1"]
            plot_line_graphs_overlayed(x_list = x_list,
                                                y_list=y_list,
                                                x_axe_name="x_axe", 
                                                y_axe_name="y_axe",
                                                legends=legends,
                                                title="Title",
                                                legend_position='upper right')
        with self.assertRaises(ValueError):

            x_list = [np.array(x1),np.array(x2)]
            y_list = [np.array(y1), np.array(y2)]
            legends = ["legend1"]
            plot_line_graphs_overlayed(x_list = x_list,
                                                y_list=y_list,
                                                x_axe_name="x_axe", 
                                                y_axe_name="y_axe",
                                                legends=legends,
                                                title="Title",
                                                legend_position='upper right')
                            
        
        x_list = [np.array(x1),np.array(x2)]
        y_list = [np.array(y1),np.array(y2)]
        legends = ["legend1","legend2"]
        plot_line_graphs_overlayed(x_list = x_list,
                                        y_list=y_list,
                                        x_axe_name="x_axe", 
                                        y_axe_name="y_axe",
                                        legends=legends,
                                        title="Title",
                                        legend_position='upper right')



    def test_simple_line_graph(self):


        with self.assertRaises(TypeError):

            x1 = np.linspace(0, 10 - 2 * 1, 200) + 1
            y1 = [1]*len(x1)

            plot_simple_line_graph(x =x1 ,
                                        y=y1,
                                        x_axe_name="x_axe", 
                                        y_axe_name="y_axe",
                                        title="Title" )

        x1 = np.linspace(0, 10 - 2 * 1, 200) + 1
        y1 = np.sin(x1) + 1.0 + 1


        plot_simple_line_graph(x =x1 ,
                                    y=y1,
                                    x_axe_name="x_axe", 
                                    y_axe_name="y_axe",
                                    title="Title")



    def test_shap_classification_summary(self):
       
        X_train,_,y_train,_ = train_test_split(*shap.datasets.iris(), test_size=0.2, random_state=0)
        shap.initjs()
        dict_map = {0 : "iris-setosa", 1 : "iris-versicolor",2 : "iris-virginica"}
        y_train = np.vectorize(dict_map.get)(y_train)
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        clf = LogisticRegression(random_state=0)
        clf.fit(X_train,y_train)
        plot_shap_classification_summary(sklearn_model=clf,X=X_train, Y=y_train,feature_names=X_train.columns,max_display=4,label_encoder=label_encoder,non_numerical_indexes=np.array([]))
        
    def test_residues(self):

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


        columns = np.array(['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad',
                            'tax', 'ptratio', 'black', 'lstat'],dtype=object)


        numerical_indexes =  np.array([0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12])
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
        X = np.concatenate((x_train, x_test), axis=0)
        target = np.concatenate((y_train, y_test), axis=0)

        plot_residues(X, target, pipeline, columns)
        
           
    def test_model_coef_weight(self):
        
        coef = np.array([-39.29539212, 56.38758691, -7.72501075, 55.14748083, 47.09622402, -224.90523448, -135.84912488, 33.26157851, 85.88832604,
                         2.82623224, 329.24528965, -90.46706709])
        columns = np.array(['Length1', 'Length2', 'Length3', 'Height', 'Width', 'Species=Bream',
                            'Species=Parkki', 'Species=Perch', 'Species=Pike', 'Species=Roach',
                            'Species=Smelt', 'Species=Whitefish'])

        plot_model_coef_weight(coef, columns)
