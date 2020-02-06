# -*- coding: utf-8 -*-
from dateutil.parser import parse

DATETIME = "DateTime"
CATEGORICAL = "Categorical"
NUMERICAL = "Numerical"


def infer_featuretypes(df, nrows=5):
    """Infer featuretypes from DataFrame columns.

    Args:
        df (pandas.DataFaame): the dataset.
        nrows (int): the number of rows to inspect.

    Returns:
        list: A list of feature types.
    """
    featuretypes = []
    for col in df.columns:
        if df.dtypes[col].kind == "O":
            if is_datetime(df[col].iloc[:nrows]):
                featuretypes.append(DATETIME)
            else:
                featuretypes.append(CATEGORICAL)
        else:
            featuretypes.append(NUMERICAL)
    return featuretypes


def is_datetime(column):
    """Returns whether a column is a DateTime."""
    for _, value in column.iteritems():
        try:
            parse(value)
            break
        except ValueError:
            return False
    return True
