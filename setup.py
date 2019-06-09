#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name="outerspace",
    version="0.1",
    packages=find_packages(),

    install_requires=[
        'opentsne~=0.3.8',
        'ipywidgets~=7.4.2',
        'bokeh~=1.2.0'
    ],
)
