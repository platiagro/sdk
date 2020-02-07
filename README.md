# PlatIAgro SDK

[![Build Status](https://travis-ci.org/platiagro/sdk.svg)](https://travis-ci.org/platiagro/sdk)
[![codecov](https://codecov.io/gh/platiagro/sdk/graph/badge.svg)](https://codecov.io/gh/platiagro/sdk)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Gitter](https://badges.gitter.im/platiagro/community.svg)](https://gitter.im/platiagro/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

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
