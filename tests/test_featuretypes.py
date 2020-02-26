# -*- coding: utf-8 -*-
from unittest import TestCase

import pandas as pd

from platiagro.featuretypes import DATETIME, CATEGORICAL, NUMERICAL, \
    infer_featuretypes, validate_featuretypes


class TestFeaturetypes(TestCase):

    def test_infer_featuretypes(self):
        df = pd.DataFrame()
        expected = []
        result = infer_featuretypes(df)
        self.assertListEqual(result, expected)

        df = pd.DataFrame({
            "col0": ["01-01-2019", "01-01-2020", "01-01-2021"],
            "col1": ["Iris-setosa", "Iris-setosa", "Iris-setosa"],
            "col2": [5.1, 4.9, 4.7],
        })
        expected = [DATETIME, CATEGORICAL, NUMERICAL]
        result = infer_featuretypes(df)
        self.assertListEqual(result, expected)

    def test_validate_featuretypes(self):
        featuretypes = []
        validate_featuretypes(featuretypes)

        featuretypes = [DATETIME, CATEGORICAL, NUMERICAL, "int"]
        with self.assertRaises(ValueError):
            validate_featuretypes(featuretypes)

        featuretypes = [DATETIME]
        validate_featuretypes(featuretypes)

        featuretypes = [CATEGORICAL]
        validate_featuretypes(featuretypes)

        featuretypes = [NUMERICAL]
        validate_featuretypes(featuretypes)