# -*- coding: utf-8 -*-
from dateutil.parser import parse

DATETIME = "DateTime"
CATEGORICAL = "Categorical"
NUMERICAL = "Numerical"


def infer_featuretypes(df, nrows=5):
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


def validate_featuretypes(featuretypes):
    """Verifies whether all feature types are valid.

    Args:
        featuretypes (list): the list of feature types.

    Raises:
        ValueError: when an invalid feature type is found."""
    valid_ones = [DATETIME, NUMERICAL, CATEGORICAL]
    if any(f not in valid_ones for f in featuretypes):
        raise ValueError("featuretype must be one of {}".format(', '.join(valid_ones)))
