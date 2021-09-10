# -*- coding: utf-8 -*-
import unittest

import numpy as np

import platiagro.pipeline


class TestPipeline(unittest.TestCase):
    def test_guarantee(self):
        """
        Should cast str to float.
        """
        x = np.array(["1", "2", "3", "4", "5"])

        gt = platiagro.pipeline.GuaranteeType()

        assert gt.fit(x).dtype == float
        assert gt.transform(x).dtype == float
        assert gt.fit_transform(x).dtype == float
        assert gt.predict(x).dtype == float
        assert gt.predict_proba(x).dtype == float
