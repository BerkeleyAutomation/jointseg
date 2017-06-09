"""
Setup of jointseg python codebase
Author: Matthew Matl
"""
from setuptools import setup

requirements = [
    'numpy',
    'matplotlib',
    'networkx',
    'cvxopt',
    'sympy',
    'scipy'
]

setup(name='jointseg',
      version='0.1.0',
      description='AutoLab mesh segmentation code',
      author='Matthew Matl',
      author_email='mmatl@berkeley.edu',
      package_dir = {'': '.'},
      packages=['jointseg'],
      install_requires=requirements,
      test_suite='test'
)

