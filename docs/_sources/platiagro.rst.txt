Functions you may use in your components:
=========================================

List, load and save datasets
----------------------------

.. currentmodule:: platiagro

.. autofunction:: list_datasets

.. code-block:: python

  from platiagro import list_datasets

  list_datasets()
  ['iris', 'boston', 'imdb']

.. autofunction:: load_dataset

.. code-block:: python

  from platiagro import load_dataset

  dataset = "iris"

  load_dataset(dataset)
          col0  col1  col2  col3  col4         col5
  0  01/01/2000   5.1   3.5   1.4   0.2  Iris-setosa
  1  01/01/2001   4.9   3.0   1.4   0.2  Iris-setosa
  2  01/01/2002   4.7   3.2   1.3   0.2  Iris-setosa
  3  01/01/2003   4.6   3.1   1.5   0.2  Iris-setosa

.. autofunction:: save_dataset

.. code-block:: python

  import pandas as pd
  from platiagro import save_dataset
  from platiagro.featuretypes import DATETIME, NUMERICAL, CATEGORICAL

  dataset = "test"

  df = pd.DataFrame({"col0": ["01/01/2000", "01/01/2001"], "col1": [1.0, -1.0], "col2": [1, 0]})
  save_dataset(dataset, df, metadata={"featuretypes": [DATETIME, NUMERICAL, CATEGORICAL]})

.. autofunction:: stat_dataset

.. code-block:: python

  from platiagro import stat_dataset

  dataset = "test"

  stat_dataset(dataset)
  {'columns': ['col0', 'col1', 'col2'], 'featuretypes': ['DateTime', 'Numerical', 'Categorical']}

.. autofunction:: download_dataset

.. code-block:: python

  from platiagro import download_dataset

  dataset = "test"
  path = "./test"

  download_dataset(dataset, path)

Load and save models
--------------------

.. currentmodule:: platiagro

.. autofunction:: load_model

.. code-block:: python

  from platiagro import load_model

  class Predictor(object):
      def __init__(self):
          self.model = load_model()

      def predict(self, X)
          return self.model.predict(X)

.. autofunction:: save_model

.. code-block:: python

  from platiagro import save_model

  model = MockModel()
  save_model(model=model)

Save metrics
------------

.. currentmodule:: platiagro

.. autofunction:: list_metrics

.. code-block:: python

  from platiagro import list_metrics

  list_metrics()
  [{'accuracy': 0.7}]

.. autofunction:: save_metrics

.. code-block:: python

  import numpy as np
  import pandas as pd
  from platiagro import save_metrics
  from sklearn.metrics import confusion_matrix

  data = confusion_matrix(y_test, y_pred, labels=labels)
  confusion_matrix = pd.DataFrame(data, columns=labels, index=labels)
  save_metrics(confusion_matrix=confusion_matrix)
  save_metrics(accuracy=0.7)
  save_metrics(reset=True, r2_score=-3.0)

List and save figures
---------------------

.. currentmodule:: platiagro

.. autofunction:: list_figures

.. code-block:: python

  from platiagro import list_figures

  list_figures()
  ['data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMAAAAC6CAIAAAB3B9X3AAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAAAG6SURBVHhe7dIxAQAADMOg+TfdicgLGrhBIBCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhDB9ho69eEGiUHfAAAAAElFTkSuQmCC']

.. autofunction:: save_figure

.. code-block:: python

  import numpy as np
  import seaborn as sns
  from platiagro import save_figure

  data = np.random.rand(10, 12)
  plot = sns.heatmap(data)
  save_figure(figure=plot.figure)

Get feature types
-----------------

.. currentmodule:: platiagro

.. autofunction:: infer_featuretypes

.. code-block:: python

  import pandas as pd
  from platiagro import infer_featuretypes

  df = pd.DataFrame({"col0": ["01/01/2000", "01/01/2001"], "col1": [1.0, -1.0], "col2": [1, 0]})
  result = infer_featuretypes(df)

.. autofunction:: validate_featuretypes

.. code-block:: python

  from platiagro import validate_featuretypes
  from platiagro.featuretypes import DATETIME, NUMERICAL, CATEGORICAL

  featuretypes = [DATETIME, NUMERICAL, CATEGORICAL]
  validate_featuretypes(featuretypes)

  featuretypes = ["float", "int", "str"]
  validate_featuretypes(featuretypes)
  ValueError: featuretype must be one of DateTime, Numerical, Categorical

Download artifact
-----------------

.. currentmodule:: platiagro

.. autofunction:: download_artifact

.. code-block:: python

  from platiagro import download_artifact

  download_artifact(name="glove_s100.zip", path="/tmp/glove_s100.zip")
  