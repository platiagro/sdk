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


    def test_residues(self):
        
        plot_residues(boston['features'],
                      boston['target'],
                      boston['regression_pipeline'], 
                      boston['features_columns'])     
           
    def test_model_coef_weight(self):
        
        coef = np.array([-39.29539212, 56.38758691, -7.72501075, 55.14748083, 47.09622402, -224.90523448, -135.84912488, 33.26157851, 85.88832604,
                         2.82623224, 329.24528965, -90.46706709])
        columns = np.array(['Length1', 'Length2', 'Length3', 'Height', 'Width', 'Species=Bream',
                            'Species=Parkki', 'Species=Perch', 'Species=Pike', 'Species=Roach',
                            'Species=Smelt', 'Species=Whitefish'])
        
        plot_model_coef_weight(coef, columns)

    def test_shap_classification_summary(self):     
            
        with self.assertWarns(expected_warning=Warning):
                    plot_shap_classification_summary(pipeline=iris['classification_pipeline'],
                                            X=iris['features'],
                                            Y=iris['target_encoded'],
                                            feature_names=iris['features_columns'],
                                            label_encoder=iris['label_encoder'],
                                            non_numerical_indexes=np.array([1]))



        plot_shap_classification_summary(pipeline=iris['classification_pipeline'],
                                            X=iris['features'],
                                            Y=iris['target_encoded'],
                                            feature_names=iris['features_columns'],
                                            label_encoder=iris['label_encoder'],
                                            non_numerical_indexes=np.array([]))

