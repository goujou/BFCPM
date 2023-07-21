from setuptools import setup, find_packages
from pathlib import Path

setup(
    name='BFMM',
    version='1.0.0',
    packages=find_packages("src"),
    package_dir={'': 'src'}
)
