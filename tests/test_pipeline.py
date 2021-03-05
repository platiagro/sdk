from unittest import TestCase
from uuid import uuid4

import numpy as np

from platiagro.pipeline import GuaranteeType


RUN_ID = str(uuid4())

class TestPipeline(TestCase):


    def setUp(self):
        pass

    
    def test_guarantee(self):

        x = np.array([1, 2, 3, 4, 5])

        gt = GuaranteeType()

        assert gt.fit(x).dtype == float
        assert gt.transform(x) == float
        assert gt.fit_transform(x) == float
        assert gt.predict(x) == float
        assert gt.predict_proba(x) == float