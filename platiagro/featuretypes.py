# -*- coding: utf-8 -*-
from dateutil.parser import parse
from typing import List

import pandas as pd

DATETIME = "DateTime"
CATEGORICAL = "Categorical"
NUMERICAL = "Numerical"


def infer_featuretypes(df: pd.DataFrame, nrows: int = 100):
    """Infer featuretypes from DataFrame columns.

    Args:
        df (pandas.DataFrame): the dataset.
        nrows (int): the number of rows to inspect.

    Returns:
        list: A list of feature types.
    """
    featuretypes = []
    for col in df.columns:
        if df.dtypes[col].kind == "O":
            if _is_number(df[col].iloc[:nrows]):
                featuretypes.append(NUMERICAL)
            elif _is_datetime(df[col].iloc[:nrows]):
                featuretypes.append(DATETIME)
            else:
                featuretypes.append(CATEGORICAL)
        else:
            featuretypes.append(NUMERICAL)
    return featuretypes


def _is_number(series: pd.Series):
    """Returns whether a series contains float values.

    Args:
        series (pandas.Series): the series.

    Returns:
        bool: True if series contains float values, otherwise False.
    """
    for _, value in series.iteritems():
        try:
            float(value)
        except ValueError:
            return False
    return True


def _is_datetime(series: pd.Series):
    """Returns whether a series contains datetime values.

    Args:
        series (pandas.Series): the series.

    Returns:
        bool: True if series contains datetime values, otherwise False.
    """
    for _, value in series.iteritems():
        try:
            parse(str(value))
            break
        except (ValueError, OverflowError):
            return False
    return True


def validate_featuretypes(featuretypes: List[str]):
    """Verifies whether all feature types are valid.

    Args:
        featuretypes (list): the list of feature types.

    Raises:
        ValueError: when an invalid feature type is found.
    """
    valid_ones = [DATETIME, NUMERICAL, CATEGORICAL]
    if any(f not in valid_ones for f in featuretypes):
        valid_str = ", ".join(valid_ones)
        raise ValueError(f"featuretype must be one of {valid_str}")
