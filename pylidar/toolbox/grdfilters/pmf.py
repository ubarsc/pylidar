"""
Functions used to implement the progressive morphological filter algorithm
(Zhang et al., 2003) to generate a raster surface which can be used to 
classify the ground returns.

Zhang, K., Chen, S., Whitman, D., Shyu, M., Yan, J., & Zhang, C. (2003). 
A progressive morphological filter for removing nonground measurements 
from airborne LIDAR data. IEEE Transactions on Geoscience and Remote Sensing, 
41(4), 872-882. 
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
import scipy
from scipy import ndimage
from scipy import interpolate

def doNearestNeighbourInterp(data, noDataMask, m, n):
    """
    Function to do nearest neighbout interpolation of filling in no data area
    """    
    # array of (number of points, 2) containing the x,y coordinates of the valid values only
    xx, yy = numpy.meshgrid(numpy.arange(data.shape[1]), numpy.arange(data.shape[0]))
    xym = numpy.vstack( (numpy.ravel(xx[noDataMask]), numpy.ravel(yy[noDataMask])) ).T
    
    # the valid values in the first band,  as 1D arrays (in the same order as their coordinates in xym)
    data0 = numpy.ravel( data[:,:][noDataMask] )
       
    # interpolator
    interp = interpolate.NearestNDInterpolator( xym, data0 )
    
    # interpolate the whole image
    result = interp(numpy.ravel(xx), numpy.ravel(yy)).reshape( xx.shape )
    return result

def elevationDiffTreshold(c, wk, wk1, s, dh0, dhmax):
    """
    Function to determine the elevation difference threshold based on window size (wk)
    c is the bin size is metres. Default values for site slope (s), initial elevation 
    differents (dh0), and maximum elevation difference (dhmax). These will change 
    based on environment.
    """
  
    if wk <= 3:
        dht = dh0
    elif wk > 3:
        dht = s * (wk-wk1) * c + dh0
    
    #However, if the difference threshold is greater than the specified max threshold, 
    #set the difference threshold equal to the max threshold    
    if dht > dhmax:
        dht == dhmax
        
    return dht                                

def disk(radius, dtype=numpy.uint8):
    """
    Generates a flat, disk-shaped structuring element.
    A pixel is within the neighborhood if the euclidean distance between
    it and the origin is no greater than radius.
    Parameters:

    * radius : int The radius of the disk-shaped structuring element.

    Other Parameters:

    * dtype : data-type The data type of the structuring element.

    Returns:

    * selem : ndarray The structuring element where elements of the neighborhood
      are 1 and 0 otherwise.

    """
    L = numpy.arange(-radius, radius + 1)
    X, Y = numpy.meshgrid(L, L)
    return numpy.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)

def doOpening(iarray, maxWindowSize, winSize1, c, s, dh0, dhmax):
    """
    A function to perform a series of iterative opening operations
    on the data array with increasing window sizes.
    """
    wkIdx = 0
    for wk in winSize1:
        #print(wk)
        if wk <= maxWindowSize:
            if wkIdx > 0:
                wk1 = winSize1[wkIdx-1]
            else:
                wk1 = 0    
            dht = elevationDiffTreshold(c, wk, wk1, s, dh0, dhmax)
            #print(dht, wk, wk1)           
            Z = iarray
            
            structureElement = disk(wk)#(int(wk), int(wk))
            #Zf = ndimage.grey_erosion(Z, size = structureElement)
            #Zf = ndimage.grey_dilation(Zf, size = structureElement)
            Zf = ndimage.morphology.grey_opening(Z, structure=structureElement, size=structureElement.shape)
            
            #Trying new method - only replace the value if it's less than the specified height
            #threshold or the zalue is less than the input
            zDiff = numpy.absolute(Z - Zf)
            iarray = numpy.where(numpy.logical_or(zDiff<=dht,Zf<Z), Zf, Z)           
            wkIdx += 1
            #print(wkIdx)        
    return iarray 


def applyPMF(dataArr, noDataMask, binGeoSize, initWinSize=1, maxWinSize=12, winSizeInc=1, slope=0.3, dh0=0.3, dhmax=5, expWinSizes=False):
    """
    Apply the progressive morphology filter (PMF) to the input data array (dataArr) 
    filtering the surface to remove regions which are not ground.
    
    * dataArr is a numpy array, usually defined as minimum Z LiDAR return within bin, on 
      which the filter is to be applied.
    * noDataMask is a numpy array specifying the cells in the input dataArr which do not
      have data (i.e., there were no points in the bin)
    * binGeoSize is the geographic (i.e., in metres) size of each square bin (i.e., 1 m)
    * initWinSize is the initial window size (Default = 1)
    * maxWinSize is the maximum window size (Default = 12)
    * winSizeInc is the increment for the window size (Default = 1)
    * slope is the slope within the scene (Default = 0.3)
    * dh0 is the initial height difference threshold for differentiating ground returns (Default = 0.3)
    * dhmax is the maximum height difference threshold for differentiating ground returns (Default = 5)
    * expWinSizes is a boolean specifying whether the windows sizes should increase exponentially or not (Default = False)
    
    Returns:
    PMF Filtered array with the same data type as the input.   

    """
    
    # Parameter k (array of values to increment the window size) k = 1,2,..., M
    k = numpy.arange(0, maxWinSize, winSizeInc)
    
    if expWinSizes is True:
        winSize = (2*initWinSize)**k + 1
    else:
        winSize = (2*k*initWinSize) + 1
    
    A = dataArr
    A = A.astype(numpy.float64)
    
    nCols = A.shape[0]
    nRows = A.shape[1]
    
    # Use nearest neighbour to interpolate cells where there is a noData value
    A = doNearestNeighbourInterp(A, noDataMask, nCols, nRows)
     
    #Create array of each window size's previous window size (i.e. window size at t-1)
    winSize_tminus1 = numpy.zeros([winSize.shape[0]])
    winSize_tminus1[1:] = winSize[:-1]
    
    # Do the opening operation and write out image file
    A = doOpening(A, maxWinSize, winSize_tminus1, binGeoSize, slope, dh0, dhmax)

    # Find the difference between the output of the opening (A) and the copy of original array (B)
    B = numpy.absolute(dataArr-A)
    
    # Create a new array (C) which takes all points from the original array copy (dataArr) within a specified 
    # threshold based on z difference (B) of the new opening array (A) - currently 0.5m
    C = numpy.where(B<=0.5, dataArr, A)

    return C.astype(numpy.float64)

