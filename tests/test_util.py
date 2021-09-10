import io
import os
import requests

import numpy as np
import pandas as pd

from category_encoders.ordinal import OrdinalEncoder
from category_encoders.one_hot import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans


def download_zipfile(name):
    if ".zip" in name:
        name = name.replace(".zip", "")

    url = f"https://raw.githubusercontent.com/platiagro/datasets/master/samples/{name}.zip"
    content = requests.get(url).content
    os.makedirs("tmp/data", exist_ok=True)
    with open(f"tmp/data/{name}.zip", "wb") as code:
        code.write(content)


def download_dataset(name):

    url = f"https://raw.githubusercontent.com/platiagro/datasets/master/samples/{name}.csv"
    content = requests.get(url).content

    dataset = pd.read_csv(io.StringIO(content.decode("utf-8")))

    return dataset


def create_classification_pipeline(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)

    numerical_indexes = np.array([0, 1, 2, 3])
    non_numerical_indexes = np.array([], int)
    ordinal_indexes_after_handle_missing_values = np.array([], int)
    one_hot_indexes_after_handle_missing_values = np.array([], int)

    pipeline = Pipeline(
        steps=[
            (
                "handle_missing_values",
                ColumnTransformer(
                    [
                        (
                            "imputer_mean",
                            SimpleImputer(strategy="mean"),
                            numerical_indexes,
                        ),
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
            (
                "estimator",
                LogisticRegression(
                    solver="liblinear",
                    penalty="l2",
                    C=1.0,
                    fit_intercept=True,
                    class_weight=None,
                    max_iter=100,
                    multi_class="auto",
                ),
            ),
        ]
    )

    _ = pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)

    return {
        "features_train": X_train,
        "features_test": X_test,
        "target_train": y_train,
        "target_test": y_test,
        "target_predicted": y_pred,
        "target_probability": y_prob,
        "classification_pipeline": pipeline,
    }


def create_clustering_pipeline(X):

    numerical_indexes = np.array([0, 1, 2, 3])
    non_numerical_indexes = np.array([], int)
    non_numerical_indexes_after_handle_missing_values = np.array([], int)

    pipeline = Pipeline(
        steps=[
            (
                "handle_missing_values",
                ColumnTransformer(
                    [
                        (
                            "imputer_mean",
                            SimpleImputer(strategy="mean"),
                            numerical_indexes,
                        ),
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
                KMeans(n_clusters=3, n_init=10, max_iter=300),
            ),
        ]
    )

    _ = pipeline.fit_transform(X)

    clusters = pipeline.named_steps.estimator.labels_

    return {"clusters": clusters, "clustering_pipeline": pipeline}


def create_regression_pipeline(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)

    numerical_indexes = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    non_numerical_indexes = np.array([], int)
    one_hot_indexes_after_handle_missing_values = np.array([], int)
    ordinal_indexes_after_handle_missing_values = np.array([], int)

    pipeline = Pipeline(
        steps=[
            (
                "handle_missing_values",
                ColumnTransformer(
                    [
                        (
                            "imputer_mean",
                            SimpleImputer(strategy="mean"),
                            numerical_indexes,
                        ),
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
            ("estimator", LinearRegression(fit_intercept=True)),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    return {
        "features_train": X_train,
        "features_test": X_test,
        "target_train": y_train,
        "target_test": y_test,
        "target_predicted": y_pred,
        "regression_pipeline": pipeline,
    }


def get_iris():

    # Default dataset
    dataset = download_dataset("iris")

    # Dataset separated from target
    X = dataset.copy()
    y = np.array(X.pop("Species"))

    # Encode target
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)

    data = {
        "dataset": dataset,
        "dataset_columns": dataset.columns,
        "features": X,
        "target": y,
        "features_columns": X.columns,
        "label_encoder": label_encoder,
        "target_encoded": y_enc,
    }

    # Create and fit pipelines
    classification_pipeline = create_classification_pipeline(X, y_enc)
    clustering_pipeline = create_clustering_pipeline(X)

    return {**data, **classification_pipeline, **clustering_pipeline}


def get_boston():

    # Default dataset
    dataset = download_dataset("boston")

    # Dataset separated from target
    X = dataset.copy()
    y = np.array(X.pop("medv"))

    data = {
        "dataset": dataset,
        "dataset_columns": dataset.columns,
        "features": X,
        "target": y,
        "features_columns": X.columns,
    }

    # Create and fit pipeline
    regression_pipeline = create_regression_pipeline(X, y)

    return {**data, **regression_pipeline}
