# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='Cregression',
    version='0.1.0',
    description='Classified Regression',
    long_description=readme,
    author='Qingzhi Ma',
    author_email='Q.Ma.2@warwick.ac.uk',
    url='https://github.com/qingzma/Cregression',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

