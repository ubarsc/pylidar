#!/usr/bin/env python

"""
Install script for PyLidar
"""
from distutils.core import setup

import pylidar

setup(name='rios',
      version=pylidar.PYLIDAR_VERSION,
      description='Raster Input/Output Simplification',
      packages=['pylidar', 'pylidar/lidarformats'],
      license='LICENSE.txt', 
      url='https://bitbucket.org/chchrsc/pylidar/overview')
      