# -*- coding: utf-8 -*-
from json import dump
from json.decoder import JSONDecodeError
from os import remove
from unittest import TestCase

from requests.exceptions import ConnectionError
from seldon_core.microservice_tester import SeldonTesterException

from platiagro import deployment


class TestDeployment(TestCase):

    def setUp(self):
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
        
        with open("invalid_contract.json", "w") as f:
            invalid_contract = {
                "features":[
                    {
                        "name":"sepal_length",
                        "dtype":"FLOAT",
                        "ftype":"FOO",
                        "range":[4,8]
                    },
                    {
                        "name":"sepal_width",
                        "dtype":"FLOAT",
                        "ftype":"continuous",
                        "range":[2,5]
                    },
                    {
                        "name":"petal_length",
                        "dtype":"FLOAT",
                        "ftype":"continuous",
                        "range":[1,10]
                    },
                    {
                        "name":"petal_width",
                        "dtype":"FLOAT",
                        "ftype":"continuous",
                        "range":[0,3]
                    }
                ],
                "targets":[
                    {
                        "name":"class",
                        "dtype":"FLOAT",
                        "ftype":"continuous",
                        "range":[0,1],
                        "repeat":3
                    }
                ]
            }
            dump(invalid_contract, f)

        with open("invalid_json.json", "w") as f:
            f.write("""{"features": [a]}""")
            f.close()

    def tearDown(self):
        remove("contract.json")
        remove("invalid_contract.json")
        remove("invalid_json.json")

    def test_test_deployment_success(self):
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
        deployment.test_deployment("contract.json")
        remove("Model.py")

    def test_test_deployment_error_on_start(self):
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
                f"        print(unknown) # intentional syntax error for test purposes\n"
                f"        self.columns_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']\n"
                f"\n"
                f"    def class_names(self):\n"
                f"        return self.columns_names\n"
                f"\n"
                f"    def predict(self, X: np.ndarray, feature_names: Iterable[str], meta: Dict = None) -> Union[np.ndarray, List, str, bytes]:\n"
                f"        return X\n"
            ))
        with self.assertRaises(ConnectionError):
            deployment.test_deployment("contract.json")
        remove("Model.py")

    def test_test_deployment_error_on_predict(self):
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
                f"        print(unknown) # intentional syntax error for test purposes\n"
                f"        return X\n"
            ))
        deployment.test_deployment("contract.json")
        remove("Model.py")

    def test_test_deployment_invalid_contract(self):
        with self.assertRaises(SeldonTesterException):
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
            deployment.test_deployment("invalid_contract.json")

    def test_test_deployment_invalid_json(self):
        with self.assertRaises(JSONDecodeError):
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
            deployment.test_deployment("invalid_json.json")
