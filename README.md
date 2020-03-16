# PlatIAgro SDK

[![Build Status](https://travis-ci.com/platiagro/sdk.svg)](https://travis-ci.com/platiagro/sdk)
[![codecov](https://codecov.io/gh/platiagro/sdk/branch/master/graph/badge.svg)](https://codecov.io/gh/platiagro/sdk)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Gitter](https://badges.gitter.im/platiagro/community.svg)](https://gitter.im/platiagro/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![Known Vulnerabilities](https://snyk.io/test/github/platiagro/sdk/badge.svg?targetFile=requirements.txt)](https://snyk.io/test/github/platiagro/sdk?targetFile=requirements.txt)

## Requirements

- [Python 3.6](https://www.python.org/downloads/)

## Quick Start

Make sure you have all requirements installed on your computer.

### Installation:

(Optional) Create a virtualenv:

```bash
virtualenv -p python3 venv
. venv/bin/activate
```

Install Python modules:

```bash
pip install .
```

## Testing

Install the testing requirements:

```bash
pip install .[testing]
```

Export these environment variables:

```bash
export MINIO_ENDPOINT=play.min.io
export MINIO_ACCESS_KEY=Q3AM3UQ867SPQQA43P2F
export MINIO_SECRET_KEY=zuf+tfteSlswRu7BJ86wekitnifILbZam1KYY3TG
```

Use the following command to run all tests:

```bash
pytest
```

## API

API Reference with examples.

### Datasets

```python
>>> from platiagro import list_datasets

>>> list_datasets()
["iris", "boston"]
```

```python
>>> from platiagro import load_dataset

>>> load_dataset(name="iris")
         col0  col1  col2  col3  col4         col5
0  01/01/2000   5.1   3.5   1.4   0.2  Iris-setosa
1  01/01/2001   4.9   3.0   1.4   0.2  Iris-setosa
2  01/01/2002   4.7   3.2   1.3   0.2  Iris-setosa
3  01/01/2003   4.6   3.1   1.5   0.2  Iris-setosa
```

```python
>>> import pandas as pd
>>> from platiagro import save_dataset
>>> from platiagro.featuretypes import DATETIME, NUMERICAL, CATEGORICAL

>>> df = pd.DataFrame({"col0": ["01/01/2000", "01/01/2001"], "col1": [1.0, -1.0], "col2": [1, 0]})
>>> save_dataset(name="test", df, metadata={"featuretypes": [DATETIME, NUMERICAL, CATEGORICAL]})
```

```python
>>> from platiagro import load_metadata

>>> load_metadata(name="test")
{'columns': ['col0', 'col1', 'col2'], 'featuretypes': ['DateTime', 'Numerical', 'Categorical']}
```

### Models

```python
>>> from platiagro import load_model

>>> load_model(experiment_id="test")
```

```python
>>> from platiagro import save_model

>>> model = MockModel()
>>> save_model(experiment_id="test", model)
```

### Metrics

```python
>>> import numpy as np
>>> import pandas as pd
>>> from platiagro import save_metrics
>>> from sklearn.metrics import confusion_matrix

>>> data = confusion_matrix(y_test, y_pred, labels=labels)
>>> confusion_matrix = pd.DataFrame(data, columns=labels, index=labels)
>>> save_metrics(experiment_id="test", confusion_matrix=confusion_matrix)
>>> save_metrics(experiment_id="test", accuracy=0.7)
>>> save_metrics(experiment_id="test", r2_score=-3.0)
```

### Figures

```python
>>> from platiagro import list_figures

>>> list_figures(experiment_id="test")
['data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMAAAAC6CAIAAAB3B9X3AAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAAAG6SURBVHhe7dIxAQAADMOg+TfdicgLGrhBIBCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhCJQCQCkQhEIhDB9ho69eEGiUHfAAAAAElFTkSuQmCC']
```

```python
>>> import numpy as np
>>> import seaborn as sns
>>> from platiagro import save_figure

data = np.random.rand(10, 12)
plot = sns.heatmap(data)
save_figure(experiment_id="test", figure=plot.figure)
```
