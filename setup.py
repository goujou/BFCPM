from pathlib import Path

from setuptools import find_packages, setup

setup(
    name="BFCPM",
    version="1.0.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
)
