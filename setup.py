import re
from os.path import dirname, join
from setuptools import find_packages, setup

with open(join(dirname(__file__), "platiagro", "__init__.py")) as fp:
    for line in fp:
        m = re.search(r'^\s*__version__\s*=\s*([\'"])([^\'"]+)\1\s*$', line)
        if m:
            version = m.group(2)
            break
    else:
        raise RuntimeError("Unable to find own __version__ string")

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

extras = {
    "plotting": [
        "shap==0.35.0",
        "seaborn>=0.10.0",
        "plotly>=4.14.1",
        "ipykernel>=5.2.0",
        "notebook>=6.0.2",
        "opencv-python==4.5.3.56",
        "unidecode==1.2.0",
        "Pillow==8.3.1"
    ],
    "metrics_nlp": [
        "nltk>=3.6.0",
        "rouge>=1.0.1",
        "bleurt @ git+https://github.com/google-research/bleurt.git",
        "jiwer>=2.2.0",
    ],
    "testing": [
        "pytest>=4.4.0",
        "pytest-xdist==1.31.0",
        "pytest-cov==2.8.1",
        "pytest-mock==3.5.1",
        "flake8==3.7.9",
    ]
}

setup(
    name="platiagro",
    version=version,
    author="Fabio Beranizo Lopes",
    author_email="fabio.beranizo@gmail.com",
    description="Python SDK for PlatIAgro.",
    license="Apache",
    url="https://github.com/platiagro/sdk",
    packages=find_packages(),
    install_requires=requirements,
    extras_require=extras,
    python_requires=">=3.7.0",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
