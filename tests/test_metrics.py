# -*- coding: utf-8 -*-
from os import environ
from unittest import TestCase

import numpy as np
import pandas as pd
from platiagro import save_metrics


class TestMetrics(TestCase):

    def test_save_metrics(self):
        with self.assertRaises(TypeError):
            save_metrics(accuracy=1.0)

        environ["EXPERIMENT_ID"] = "test"
        with self.assertRaises(TypeError):
            save_metrics(accuracy=1.0)

        environ["OPERATOR_ID"] = "test"
        save_metrics(accuracy=1.0)

        del environ["EXPERIMENT_ID"]
        del environ["OPERATOR_ID"]

        with self.assertRaises(TypeError):
            save_metrics(experiment_id="test", operator_id="test",
                         accuracy=lambda x: "WUT")

        data = np.array([[2, 0, 0],
                         [0, 0, 1],
                         [1, 0, 2]])
        labels = ["Iris-setosa", "Iris-virginica", "Iris-versicolor"]
        confusion_matrix = pd.DataFrame(data, columns=labels, index=labels)
        save_metrics(experiment_id="test", operator_id="test",
                     confusion_matrix=confusion_matrix)
        save_metrics(experiment_id="test", operator_id="test", reset=True,
                     r2_score=-3.0)
