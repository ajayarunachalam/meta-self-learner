#!/usr/bin/env python
'''
__author__ = 'Ajay Arunachalam'
__version__ = '0.0.1'
__date__ = '25.4.2021'
'''

import sys

from setuptools import find_packages, setup


def setup_package():
    metadata = dict(
        name='meta-self-learner',
        version='0.0.1',
        description='Meta Ensemble Self-Learning with Optimization Objective Functions',
        author='Ajay Arunachalam',
        author_email='ajay.arunachalam08@gmail.com',
        license='GNU General Public License',
        url='https://github.com/ajayarunachalam/meta-self-learner',
        packages=find_packages(),
        long_description=open('./README.rst').read(),
        long_description_content_type="text/x-rst",
        install_requires=open('./requirements.txt').read().split()
    )

    setup(**metadata)


if (__name__ == '__main__'):
    setup_package()
