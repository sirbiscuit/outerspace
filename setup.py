#!/usr/bin/env python
from setuptools import setup, find_packages

exec(open('outerspace/version.py').read())

setup(
    name='outerspace',
    version=__version__,
    description='',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/sirbiscuit/outerspace',
    author='sirbiscuit',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'opentsne~=0.3.11',
        'ipywidgets~=7.5.1',
        'bokeh~=1.2.0',
    ],
    extras_require={ 
        'test': ['pytest'] 
    }
)
