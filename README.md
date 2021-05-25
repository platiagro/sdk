# PlatIAgro SDK

[![Build Status](https://github.com/platiagro/sdk/workflows/Python%20application/badge.svg)](https://github.com/platiagro/sdk/actions?query=workflow%3A%22Python+application%22)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=platiagro_sdk&metric=alert_status)](https://sonarcloud.io/dashboard?id=platiagro_sdk)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Requirements

- [Python 3.7](https://www.python.org/downloads/)

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
export MINIO_ENDPOINT=localhost:9000
export MINIO_ACCESS_KEY=minio
export MINIO_SECRET_KEY=minio123
export PROJECTS_ENDPOINT=projects.platiagro:8080
```

(Optional) Start a MinIO instance:

```bash
docker run -d -p 9000:9000 \
  --name minio \
  --env "MINIO_ACCESS_KEY=$MINIO_ACCESS_KEY" \
  --env "MINIO_SECRET_KEY=$MINIO_SECRET_KEY" \
  minio/minio:RELEASE.2018-02-09T22-40-05Z \
  server /data
```

Use the following command to run all tests:

```bash
pytest
```

Use the following command to run lint:

```bash
flake8 --max-line-length 127 platiagro/
```

## API

See the [PlatIAgro SDK API doc](https://platiagro.github.io/sdk/) for API specification.

## Update Documentation

After making some changes in PlatIAgro SDK, you need to update the docs in this [file](https://github.com/platiagro/sdk/blob/master/docs/source/platiagro.rst) and run these commands:

```bash
pip install sphinx
cd docs/
make html
rm -r _static/ _sources/
mv build/html/* .
```