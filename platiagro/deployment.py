# -*- coding: utf-8 -*-
"""A module for testing components before deployment."""
from os import environ
from random import randint
from subprocess import PIPE, Popen
from typing import Optional

from requests import Session
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from .util import get_experiment_id, get_operator_id


def test_deployment(contract: str,
                    module: str = "Model",
                    interface_name: str = "REST",
                    service_type: str = "MODEL",
                    parameters: str = "[]",
                    log_level: str = "DEBUG",
                    port: Optional[int] = None):
    """Starts a service wrapping a Model, then sends a request to the service.

    Args:
        contract (str): Path to the file defining the data you intend to send in a request and the response you expect back.
        module (str, optional): Name of the Python module. Defaults to "Model".
        interface_name (str, optional): Name of the API interface. Defaults to "REST".
        service_type (str, optional): Type of the component. Defaults to "MODEL".
        parameters (str, optional): Parameters (in JSON format). Defaults to "[]".
        log_level (str, optional): Log level. Defaults to "DEBUG".
        port (int, optional): Port for HTTP server. Default is randomized betwee 5000 and 9000.
    """
    # gets experiment_id and operator_id from notebook metadata
    experiment_id = get_experiment_id(raise_for_none=False, default="")
    operator_id = get_operator_id(raise_for_none=False, default="")

    if port is None:
        port = randint(5000, 9000)

    # start HTTP server
    cmd = (
        f"seldon-core-microservice "
        f"{module} "
        f"{interface_name} "
        f"--service-type "
        f"{service_type} "
        f"--parameters "
        f"'{parameters}' "
        f"--log-level "
        f"{log_level}"
    )
    env = environ.copy()
    env.update({
        "EXPERIMENT_ID": experiment_id,
        "OPERATOR_ID": operator_id,
        "PREDICTIVE_UNIT_SERVICE_PORT": f"{port}",
    })
    pserver = Popen(
        cmd,
        env=env,
        shell=True,
        stdout=PIPE,
    )

    # verify the process is up and running
    retry_strategy = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        method_whitelist=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    sess = Session()
    sess.mount("http://", adapter)
    sess.get(f"http://localhost:{port}/health/ping")

    # start a process to test the server
    cmd = (
        f"seldon-core-microservice-tester "
        f"{contract} "
        f"localhost "
        f"{port} "
        f"--endpoint "
        f"predict "
        f"-p"
    )
    pclient = Popen(
        cmd,
        shell=True,
        stdout=PIPE,
    )

    for line in pclient.stdout:
        print(line.decode(), end="")

    pserver.stdout.close()
    pclient.stdout.close()
    pserver.terminate()
