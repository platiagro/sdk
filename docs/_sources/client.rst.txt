
Client
======
  You may access PlatIAgro Projects API using the following functions.

.. currentmodule:: platiagro.client

Experiments
----------------

.. autofunction:: list_experiments

.. code-block:: python

  from platiagro.client import list_experiments
  
  project_id = 'c862813f-13e8-4adb-a430-56a1444209a0'

  list_experiments(project_id)
  {'experiments': [{'uuid': 'a8ab15b1-7a90-4f18-b18d-e14f7422c938', 'name': 'foo', 'project_id': 'c862813f-13e8-4adb-a430-56a1444209a0'}]}

.. autofunction:: get_experiment_by_name

.. code-block:: python

  from platiagro.client import get_experiment_by_name

  project_id = 'c862813f-13e8-4adb-a430-56a1444209a0'
  experiment_name = 'foo'

  get_experiment_by_name(project_id, experiment_name)
  {'uuid': 'a8ab15b1-7a90-4f18-b18d-e14f7422c938', 'name': 'foo', 'project_id': 'c862813f-13e8-4adb-a430-56a1444209a0'}

.. autofunction:: run_experiment

.. code-block:: python

  from platiagro.client import run_experiment

  project_name = 'foo'
  experiment_name = 'bar'

  run_experiment(experiment_name, project_name)
  {'uuid': 'bc4a0874-4a6b-4e20-bd7e-ed00c51fd8ea', 'createdAt': '2021-05-24T13:57:20.767Z', 'operators': []}

Deployments
----------------

.. autofunction:: list_deployments

.. code-block:: python

  from platiagro.client import list_deployments

  project_id = 'c862813f-13e8-4adb-a430-56a1444209a0'

  list_deployments(project_id)
  {'deployments': [{'uuid': 'ea5661ab-28fe-4531-b310-8a37bd77197b', 'name': 'foo', 'project_id': 'c862813f-13e8-4adb-a430-56a1444209a0'}]}

.. autofunction:: get_deployment_by_name

.. code-block:: python

  from platiagro.client import get_deployment_by_name

  project_id = 'c862813f-13e8-4adb-a430-56a1444209a0'
  deployment_name = 'foo'

  get_deployment_by_name(project_id, deployment_name)
  {'uuid': 'ea5661ab-28fe-4531-b310-8a37bd77197b', 'name': 'foo', 'project_id': 'c862813f-13e8-4adb-a430-56a1444209a0'}

.. autofunction:: run_deployment

.. code-block:: python

  from platiagro.client import run_deployment

  project_name = 'foo'
  deployment_name = 'bar'

  run_deployment(deployment_name, project_name)
  {'uuid': 'b793c517-2f95-41f0-a4bb-25da50d6e052', 'createdAt': '2021-05-24T13:57:20.767Z', 'operators': []}

Projects
--------

.. autofunction:: list_projects

.. code-block:: python

  from platiagro.client import list_projects

  list_projects()
  {'projects': [{'uuid': 'c862813f-13e8-4adb-a430-56a1444209a0', 'name': 'foo'}]}

.. autofunction:: get_project_by_name

.. code-block:: python

  from from platiagro.client import get_project_by_name

  project_name = 'foo'
  get_project_by_name(project_name)
  {'uuid': 'c862813f-13e8-4adb-a430-56a1444209a0', 'name': 'foo'}

Tasks
-----

.. autofunction:: create_task

.. code-block:: python

  from platiagro.client import create_task

  create_task(name = "foo")
  {'uuid': '3fa85f64-5717-4562-b3fc-2c963f66afa6', 'name': 'foo', 'createdAt': '2021-05-24T14:04:44.126Z', 'updatedAt': '2021-05-24T14:04:44.126Z'}
