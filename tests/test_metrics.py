# -*- coding: utf-8 -*-
from unittest import TestCase

import numpy as np
import pandas as pd
from platiagro import save_metrics


class TestMetrics(TestCase):

    def test_save_metrics(self):
        with self.assertRaises(TypeError):
            save_metrics(experiment_id="test", accuracy=lambda x: "WUT")

        data = np.array([[2, 0, 0],
                         [0, 0, 1],
                         [1, 0, 2]])
        labels = ["Iris-setosa", "Iris-virginica", "Iris-versicolor"]
        confusion_matrix = pd.DataFrame(data, columns=labels, index=labels)
        save_metrics(experiment_id="test", confusion_matrix=confusion_matrix)
