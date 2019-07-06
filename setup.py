#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name='outerspace',
    version='0.1.0',
    description='',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/sirbiscuit/outerspace',
    author='sirbiscuit',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'opentsne~=0.3.8',
        'ipywidgets~=7.4.2',
        'bokeh~=1.2.0',
        'sharedmem~=0.3.6'
    ],
)
