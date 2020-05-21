# -*- coding: utf-8 -*-
from json import dump
from os import remove
from unittest import TestCase

from platiagro import deployment


class TestDeployment(TestCase):

    def setUp(self):
        with open("Model.py", "w") as f:
            f.write((
                f"import logging\n"
                f"from typing import List, Iterable, Dict, Union\n"
                f"\n"
                f"import numpy as np\n"
                f"\n"
                f"logger = logging.getLogger(__name__)\n"
                f"\n"
                f"\n"
                f"class Model(object):\n"
                f"    def __init__(self, dataset: str = None, target: str = None):\n"
                f"        self.columns_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']\n"
                f"\n"
                f"    def class_names(self):\n"
                f"        return self.columns_names\n"
                f"\n"
                f"    def predict(self, X: np.ndarray, feature_names: Iterable[str], meta: Dict = None) -> Union[np.ndarray, List, str, bytes]:\n"
                f"        return X\n"
            ))

        with open("contract.json", "w") as f:
            contract = {
                "features": [
                    {
                        "name": "sepal_length",
                        "dtype": "FLOAT",
                        "ftype": "continuous",
                        "range": [4, 8],
                    },
                    {
                        "name": "sepal_width",
                        "dtype": "FLOAT",
                        "ftype": "continuous",
                        "range": [2, 5],
                    },
                    {
                        "name": "petal_length",
                        "dtype": "FLOAT",
                        "ftype": "continuous",
                        "range": [1, 10],
                    },
                    {
                        "name": "petal_width",
                        "dtype": "FLOAT",
                        "ftype": "continuous",
                        "range": [0, 3],
                    },
                ],
                "targets": [
                    {
                        "name": "sepal_length",
                        "dtype": "FLOAT",
                        "ftype": "continuous",
                        "range": [4, 8],
                    },
                    {
                        "name": "sepal_width",
                        "dtype": "FLOAT",
                        "ftype": "continuous",
                        "range": [2, 5],
                    },
                    {
                        "name": "petal_length",
                        "dtype": "FLOAT",
                        "ftype": "continuous",
                        "range": [1, 10],
                    },
                    {
                        "name": "petal_width",
                        "dtype": "FLOAT",
                        "ftype": "continuous",
                        "range": [0, 3],
                    },
                ]
            }
            dump(contract, f)

    def tearDown(self):
        remove("contract.json")
        remove("Model.py")

    def test_test_deployment(self):
        deployment.test_deployment("contract.json")