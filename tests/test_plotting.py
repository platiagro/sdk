from unittest import TestCase
from uuid import uuid4

import numpy as np
import pandas as pd

from plotting import plot_roc_curve

RUN_ID = str(uuid4())

class TestPlotting(TestCase):

    def setUp(self):
        
        self.data = pd.DataFrame([[1, 0.2, 0.5],
                                [2, 0.4, 1.0],
                                [3, 0.8, 2.0],
                                [4, 1.0, 4.0],
                                [1, 0.2, 0.5],
                                [2, 0.4, 1.0],
                                [3, 0.8, 2.0],
                                [4, 1.0, 4.0],
                                [1, 0.2, 0.5],
                                [2, 0.4, 1.0],
                                [3, 0.8, 2.0],
                                [4, 1.0, 4.0],
                                [1, 0.2, 0.5],
                                [2, 0.4, 1.0],
                                [3, 0.8, 2.0],
                                [4, 1.0, 4.0]])

        self.y_class = np.array(['a', 'b', 'c', 'd',
                        'a', 'b', 'c', 'd',
                        'a', 'b', 'c', 'd',
                        'a', 'b', 'c', 'd'])
        
        self.y_class_pred = np.array([0.8, 0.1, 0.1, 0.0,
                            0.2, 0.7, 0.1, 0.0,
                            0.0, 0.1, 0.8, 0.1,
                            0.0, 0.0, 0.0, 1.0,
                            0.8, 0.1, 0.1, 0.0,
                            0.2, 0.7, 0.1, 0.0,
                            0.0, 0.1, 0.8, 0.1,
                            0.0, 0.0, 0.0, 1.0,
                            0.8, 0.1, 0.1, 0.0,
                            0.2, 0.7, 0.1, 0.0,
                            0.0, 0.1, 0.8, 0.1,
                            0.0, 0.0, 0.0, 1.0,
                            0.8, 0.1, 0.1, 0.0,
                            0.2, 0.7, 0.1, 0.0,
                            0.0, 0.1, 0.8, 0.1,
                            0.0, 0.0, 0.0, 1.0])
        
        self.y_reg = np.array([0.2, 0.4, 0.6, 0.8,
                    0.2, 0.4, 0.6, 0.8,
                    0.2, 0.4, 0.6, 0.8,
                    0.2, 0.4, 0.6, 0.8])
        
        self.y_reg_pred = np.array([0.25, 0.35, 0.65, 0.85,
                    0.35, 0.35, 0.60, 0.8,
                    0.15, 0.4, 0.44, 0.75,
                    0.2, 0.4, 0.65, 0.7])

    def test_roc(self):

        # plot_roc_curve(self.y_class, self.y_class_pred, list(set(self.y_class)))
        pass
