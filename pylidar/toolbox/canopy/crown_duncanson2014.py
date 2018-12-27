"""
Functions for multilayer crown delineation from ALS point cloud data (Duncanson et al., 2014)
Assumes a ground classification exists and is included in the input file format.
"""

# This file is part of PyLidar
# Copyright (C) 2015 John Armston, Pete Bunting, Neil Flood, Sam Gillingham
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import print_function, division

import numpy
from numba import jit

from pylidar import lidarprocessor


def run_crown_duncanson2014(dataFiles, controls, otherargs, outfiles):
    """
    Main function for CROWN_DUNCANSON2014
    """
    pass

    
def makeCHM(smoother=1):
    """
    
    """

    
def applyWatershed():
    """
    
    """

@jit
def extractCrownMetrics():
    """
    
    """
    