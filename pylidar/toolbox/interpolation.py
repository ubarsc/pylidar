"""
Functions which can be used to perform interpolation of point data
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

import os
import numpy

# See which interpolator we have access to
haveScipyInterp = True
try:
    import scipy.interpolate
except ImportError:
    haveScipyInterp = False

# https://bitbucket.org/chchrsc/cgalinterp
# Have had trouble with CGAL on some datasets
# Current preference is to use PyNNInterp (below)
# But we have left support in just in case.
haveCGALInterpPy = True
try:
    import cgalinterp
except ImportError as cgalInterpErr:
    haveCGALInterpPy = False
    
# Preferred interpolator - see https://bitbucket.org/petebunting/pynninterp
havePyNNInterp = True
try:
    import pynninterp
except ImportError:
    havePyNNInterp = False

# Exception type for this module
class InterpolationError(Exception):
    "Interpolation Error"
    
def interpGrid(xVals, yVals, zVals, gridCoords, method='pynn'):
    """
    A function to interpolate values to a regular gridCoords given 
    an irregular set of input data points
    
    * xVals is an array of X coordinates for known points
    * yVals is an array of Y coordinates for known points
    * zVals is an array of values associated with the X,Y points to be interpolated
    * gridCoords is a 2D array with the X,Y values for each 'pixel' in the grid; use data.info.getBlockCoordArrays()
    * method is a string specifying the method of interpolation to use, 'nearest', 'linear', 'cubic', 'cgalnn', 'pynn', 'pylinear'
    
    returns grid of float64 values with the same dimensions are the gridCoords with interpolated Z values.
    """
    
    if (xVals.shape != yVals.shape) & (yVals.shape != zVals.shape):
        raise InterpolationError("X, Y and Z inputs did not have the same shapes.")
    
    if xVals.shape[0] < 4:
        raise InterpolationError("Must have at least 4 input points to create interpolator")
    
    if method == 'nearest' or method == 'linear' or method == 'cubic':
        if not haveScipyInterp:
            msg = "scipy must be installed for nearest, linear or cubic"
            raise InterpolationError(msg)
        interpZ = scipy.interpolate.griddata((xVals, yVals), zVals, (gridCoords[0].flatten(), gridCoords[1].flatten()), method=method, rescale=True)
        interpZ = interpZ.astype(numpy.float64)
        out = numpy.reshape(interpZ, gridCoords[0].shape)

    elif method == 'cgalnn':
        if not haveCGALInterpPy:
            msg = "The cgalinterp python bindings required for natural neighbour interpolation and could not be imported"
            raise InterpolationError(msg)
        xVals = xVals.astype(numpy.float64)
        yVals = yVals.astype(numpy.float64)
        zVals = zVals.astype(numpy.float64)
        try:
            out = cgalinterp.NaturalNeighbour(xVals, yVals, zVals, gridCoords[0], gridCoords[1])
        except Exception as e:
            # rethrow cgalinterp exception type so it can be more easily caught
            raise InterpolationError(str(e))

    elif method == 'pynn':
        if not havePyNNInterp:
            msg = "The pynninterp python bindings required for natural neighbour interpolation and could not be imported"
            raise InterpolationError(msg)
        xVals = xVals.astype(numpy.float64)
        yVals = yVals.astype(numpy.float64)
        zVals = zVals.astype(numpy.float64)
        if isinstance(gridCoords, numpy.ndarray):
            gridCoords = gridCoords.astype(numpy.float64)
        else:
            gridCoords = (gridCoords[0].astype(numpy.float64),
                            gridCoords[1].astype(numpy.float64))
            
        try:
            out = pynninterp.NaturalNeighbour(xVals, yVals, zVals, gridCoords[0], gridCoords[1])
        except Exception as e:
            # rethrow pynninterp exception type so it can be more easily caught
            raise InterpolationError(str(e))

    elif method == 'pylinear':
        if not havePyNNInterp:
            msg = "The pynninterp python bindings required for linear (TIN?) interpolation and could not be imported"
            raise InterpolationError(msg)
        xVals = xVals.astype(numpy.float64)
        yVals = yVals.astype(numpy.float64)
        zVals = zVals.astype(numpy.float64)
        if isinstance(gridCoords, numpy.ndarray):
            gridCoords = gridCoords.astype(numpy.float64)
        else:
            gridCoords = (gridCoords[0].astype(numpy.float64),
                            gridCoords[1].astype(numpy.float64))

        try:
            out = pynninterp.Linear(xVals, yVals, zVals, gridCoords[0], gridCoords[1])
        except Exception as e:
            # rethrow pynninterp exception type so it can be more easily caught
            raise InterpolationError(str(e))

    else:
        raise InterpolationError("Interpolaton method '%s' was not recognised" % method)

    return out

def interpPoints(xVals, yVals, zVals, ptCoords, method='pynn'):
    """
    A function to interpolate values to a set of points given 
    an irregular set of input data points
    
    * xVals is an array of X coordinates for known points
    * yVals is an array of Y coordinates for known points
    * zVals is an array of values associated with the X,Y points to be interpolated
    * ptCoords is a 2D array with pairs of xy values (shape: N*2)
    * method is a string specifying the method of interpolation to use, 'pynn' is the only currently implemented one.
    
    returns 1d array with Z values.
    """
    if method == 'pynn':
        if not havePyNNInterp:
            msg = "The nninterp python bindings required for natural neighbour interpolation and could not be imported"
            raise InterpolationError(msg)

        out = pynninterp.NaturalNeighbourPts(xVals, yVals, zVals, ptCoords)

    elif method == 'cgalnn':
        if not haveCGALInterpPy:
            msg = "The cgalinterp python bindings required for natural neighbour interpolation and could not be imported"
            raise InterpolationError(msg)

        out = cgalinterp.NaturalNeighbourPts(xVals, yVals, zVals, ptCoords)

    else:
        raise InterpolationError("Interpolaton method '%s' was not recognised" % method)

    return out
