"""
Functions which can be used to perform interpolation of point data
"""
# This file is part of PyLidar
# Copyright (C) 2015 John Armston, Neil Flood, Sam Gillingham and Pete Bunting
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
import scipy.interpolate

def interpGrid(xVals, yVals, zVals, gridCoords, method, checkPtDist=True):
    """
    A function to interpolate values to a regular gridCoords given 
    an irregular set of input data points
    
    * xVals is an array of X coordinates for known points
    * yVals is an array of Y coordinates for known points
    * zVals is an array of values associated with the X,Y points to be interpolated
    * gridCoords is a 2D array with the X,Y values for each 'pixel' in the grid; use data.info.getBlockCoordArrays()
    * method is a string specifying the method of interpolation to use, 'nearest', 'linear', 'cubic', 'nn'
    
    returns grid of float64 values with the same dimensions are the gridCoords with interpolated Z values.
    """
    
    if (xVals.shape != yVals.shape) & (yVals.shape != zVals.shape):
        raise Exception("X, Y and Z inputs did not have the same shapes.")
    
    if xVals.shape[0] < 4:
        raise Exception("Must have at least 4 input points to create interpolator")
    
    if checkPtDist:
        if (xVals.shape[0] < 100) & (numpy.var(xVals) < 4.0) & (numpy.var(yVals) < 4.0):
            raise Exception("Both the X and Y input coordinates must have a variance > 4 when the number of points is < 100.")
        
    if method == 'nearest' or method == 'linear' or method == 'cubic':
        interpZ = scipy.interpolate.griddata((xVals, yVals), zVals, (gridCoords[0].flatten(), gridCoords[1].flatten()), method=method, rescale=True)
        interpZ = interpZ.astype(numpy.float64)
        out = numpy.reshape(interpZ, gridCoords[0].shape)
    elif method == 'rbf':
        raise Exception("RBF has not been tested and may well have issues and further options need to be explored...")
        rbfInterp = scipy.interpolate.Rbf(xVals, yVals, zVals)
        interpZ = rbfInterp(gridCoords[0].flatten(), gridCoords[1].flatten())
        interpZ = interpZ.astype(numpy.float64)
        out = numpy.reshape(interpZ, gridCoords[0].shape)
    elif method == 'nn':
        raise Exception("Natural Neighbour interpolation is not ready yet...")
    else:
        raise Exception("Interpolaton method was not recognised")
    return out
