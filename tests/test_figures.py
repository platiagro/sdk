# -*- coding: utf-8 -*-
from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np
from platiagro import list_figures, save_figure


class TestFigures(TestCase):

    def test_list_figures(self):
        result = list_figures(experiment_id="test")
        self.assertTrue(isinstance(result, list))

    def test_save_figure(self):
        with self.assertRaises(TypeError):
            save_figure(experiment_id="test", figure="path/to/figure")

        t = np.arange(0.0, 2.0, 0.01)
        s = 1 + np.sin(2 * np.pi * t)
        fig, ax = plt.subplots()
        ax.plot(t, s)
        save_figure(experiment_id="test", figure=fig)
