# -*- coding: utf-8 -*-
import unittest

import pandas as pd

import platiagro


class TestFeaturetypes(unittest.TestCase):
    def test_infer_featuretypes_empty(self):
        """
        Should return empty list when dataframe is empty.
        """
        df = pd.DataFrame()

        result = platiagro.infer_featuretypes(df=df)

        expected = []
        self.assertListEqual(result, expected)

    def test_infer_featuretypes_not_empty(self):
        """
        Should return list of features when dataframe is not empty.
        """
        df = pd.DataFrame(
            {
                "col0": ["01-01-2019", "01-01-2020", "01-01-2021"],
                "col1": ["Iris-setosa", "Iris-setosa", "Jun 999999999999999"],
                "col2": [5.1, float("nan"), 4.7],
                "col3": ["22.4", "10", "4.7"],
            }
        )

        result = platiagro.infer_featuretypes(df=df)

        expected = [
            platiagro.DATETIME,
            platiagro.CATEGORICAL,
            platiagro.NUMERICAL,
            platiagro.NUMERICAL,
        ]
        self.assertListEqual(result, expected)

    def test_validate_featuretypes_empty(self):
        """
        Should not raise exception when feature list is empty.
        """
        featuretypes = []

        platiagro.validate_featuretypes(featuretypes)

    def test_validate_featuretypes_value_error(self):
        """
        Should raise exception when feature list has invalid value (eg. "int").
        """
        featuretypes = [
            platiagro.DATETIME,
            platiagro.CATEGORICAL,
            platiagro.NUMERICAL,
            "int",
        ]
        with self.assertRaises(ValueError):
            platiagro.validate_featuretypes(featuretypes)
