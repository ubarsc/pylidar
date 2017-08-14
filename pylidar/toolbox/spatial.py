"""
Utility functions for assisting with Spatial Processing
in Pylidar. 
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
from osgeo import osr
from osgeo import gdal
from rios import imageio
from rios import calcstats
from rios import pixelgrid
from pylidar import userclasses
from pylidar import lidarprocessor # for DEFAULT_RASTERDRIVERNAME etc. Maybe should move?
from pylidar.lidarformats import generic

gdal.UseExceptions()

class SpatialException(generic.LiDARFileException):
    """
    An exception that is raised by this module
    """
    pass

def xyToRowCol(x, y, xMin, yMax, pixSize):
    """
    For the given pixel size and xMin, yMax, convert the
    given arrays of x and y into arrays of row and column in a regular grid 
    across the tile. 

    xMin and yMax represent the top/left corner of the top/left pixel of the image
    
    Assumes that the bounds of the grid are to be fixed on integer coordinates. 
    
    Return a tuple of arrays 
        (row, col)
    
    """
    col = ((x - xMin) / pixSize).astype(numpy.uint)
    row = ((yMax - y) / pixSize).astype(numpy.uint)
    return (row, col)

@jit
def xyToRowColNumba(x, y, xMin, yMax, pixSize):
    """
    Same as xyToRowCol but jitted so can be called from inside Numba
    """
    col = int((x - xMin) / pixSize)
    row = int((yMax - y) / pixSize)
    return (row, col)

def readLidarPoints(filename, classification=None, boundingbox=None, 
            colNames=['X', 'Y', 'Z']):
    """
    Read the requested columns for the points in the given file (or files if 
    filename is a list), in a memory-efficient manner. 
    Uses pylidar to read only a block of points at a time, and select out just the 
    desired columns. When the input file is a .las file, this saves quite a lot
    of memory, in comparison to reading in all points at once, since all columns for all points
    have to be read in at the same time. 
    
    Optionally filter by CLASSIFICATION column with a value from the generic.CLASSIFICATION_*
    constants.
    
    If boundingbox is given, it is a tuple of
    (xmin, xmax, ymin, ymax)
    and only points within this box are included. 
    
    Return a single recarray with only the selected columns, and only the selected points. 
    
    """
    datafiles = lidarprocessor.DataFiles()
    # could be a list
    datafiles.infile = lidarprocessor.LidarFile(filename, lidarprocessor.READ)
    
    otherargs = lidarprocessor.OtherArgs()
    otherargs.classification = classification
    otherargs.colNames = colNames
    otherargs.dataArrList = []
    otherargs.boundingbox = boundingbox
    
    controls = lidarprocessor.Controls()
    controls.setSpatialProcessing(False)
    
    lidarprocessor.doProcessing(selectColumns, datafiles, otherArgs=otherargs, controls=controls)
    
    # Put all the separate rec-arrays together
    nPts = sum([len(a) for a in otherargs.dataArrList])
    if nPts > 0:
        fullArr = numpy.zeros(nPts, dtype=otherargs.dataArrList[0].dtype)
        i = 0
        for dataArr in otherargs.dataArrList:
            numPts = len(dataArr)
            fullArr[i:i+numPts] = dataArr
            i += len(dataArr)
    else:
        fullArr = numpy.array([])
    
    return fullArr

def selectColumns(data, otherargs):
    """
    Called from pylidar's doProcessing. 
    
    Read the next block of lidar points, select out the requested columns. If requested, 
    filter to ground only. If requested, restrict to the given bounding box. 
    
    """
    # have to deal with data.infile potentially being a list
    if isinstance(data.infile, userclasses.LidarData):
        driverList = [data.infile]
    else:
        driverList = data.infile

    for driver in driverList:

        dataArr = driver.getPoints(colNames=otherargs.colNames)

        if otherargs.classification is not None:
            if 'CLASSIFICATION' in otherargs.colNames:
                pntClass = data['CLASSIFICATION']
            else:
                pntClass = driver.getPoints(colNames='CLASSIFICATION')
            mask = (pntClass == otherargs.classification)
            dataArr = dataArr[mask]
    
        if otherargs.boundingbox is not None:
            (xmin, xmax, ymin, ymax) = otherargs.boundingbox
            if 'X' in otherargs.colNames:
                x = dataArr['X']
            else:
                x = driver.getPoints(colNames='X')
            if 'Y' in otherargs.colNames:
                y = dataArr['Y']
            else:
                y = driver.getPoints(colNames='Y')

            # !!!!!!!!!!!!!!!!!!
            # Now same expression as the spatial index stuff, not sure 
            # if this is correct but means will only get points once...
            mask = ((x >= xmin) & (x < xmax) & (y > ymin) & (y <= ymax))
            dataArr = dataArr[mask]

        if len(dataArr) > 0:
            # Stash in the list of arrays. 
            otherargs.dataArrList.append(dataArr)

def getGridInfoFromData(xdata, ydata, binSize):
    """
    Given an array of X coords, an array of Y coords,
    plus a binSize return a tuple of (xMin, yMax, ncols, nrows)
    for doing operations on a grid
    """
    xMin = xdata.min()
    yMax = ydata.max()
    # nasty rounding errors propogated with ceil()
    ncols = int(numpy.round((xdata.max() - xMin) / binSize))
    nrows = int(numpy.round((yMax - ydata.min()) / binSize))
    return (xMin, yMax, ncols, nrows)

def getGridInfoFromHeader(header, binSize, footprint=lidarprocessor.UNION):
    """
    Given a Lidar file header (or a list of headers - maximum extent 
    will be calculated)
    plus a binSize return a tuple of (xMin, yMax, ncols, nrows)
    for doing operations on a grid
    Specify lidarprocessor.UNION or lidarprocessor.INTERSECTION to determine
    how multiple headers are combined.
    """
    if isinstance(header, dict):
        headers = [header]
    else:
        headers = header

    # get the dims of the first one
    pixGrid = pixelgrid.PixelGridDefn(xMin=headers[0]['X_MIN'],
                xMax=headers[0]['X_MAX'], yMax=headers[0]['Y_MAX'],
                yMin=headers[0]['Y_MIN'], xRes=binSize, yRes=binSize)

    for header in headers[1:]:
        newGrid = pixelgrid.PixelGridDefn(xMin=header['X_MIN'],
                xMax=header['X_MAX'], yMax=header['Y_MAX'],
                yMin=header['Y_MIN'], xRes=binSize, yRes=binSize)

        if footprint == lidarprocessor.UNION:
            pixGrid = pixGrid.union(newGrid)
        elif footprint == lidarprocessor.INTERSECTION:
            pixGrid = pixGrid.intersection(newGrid)
        else:
            msg = 'unsupported footprint value'
            raise SpatialException(msg)

    # nasty rounding errors propogated with ceil()
    ncols = int(numpy.round((pixGrid.xMax - pixGrid.xMin) / binSize))
    nrows = int(numpy.round((pixGrid.yMax - pixGrid.yMin) / binSize))
    return (pixGrid.xMin, pixGrid.yMax, ncols, nrows)

def getBlockCoordArrays(xMin, yMax, nCols, nRows, binSize):
    """
    Return a tuple of the world coordinates for every pixel
    in the current block. Each array has the same shape as the 
    current block. Return value is a tuple::

        (xBlock, yBlock)

    where the values in xBlock are the X coordinates of the centre
    of each pixel, and similarly for yBlock. 
                                                    
    The coordinates returned are for the pixel centres. This is 
    slightly inconsistent with usual GDAL usage, but more likely to
    be what one wants.         
        
    """
    # create the indices
    (rowNdx, colNdx) = numpy.mgrid[0:nRows, 0:nCols]
    xBlock = (xMin + binSize/2.0 + colNdx * binSize)
    yBlock = (yMax - binSize/2.0 - rowNdx * binSize)
    return (xBlock, yBlock)    

def readImageLayer(inFile, layerNum=1):
    """
    Read a layer from a GDAL supported dataset and return it as 
    a 2d numpy array along with georeferencing information.

    Returns a tuple with (data, xMin, yMax, binSize)
    """
    ds = gdal.Open(inFile)
    band = ds.GetRasterBand(layerNum)
    data = band.ReadAsArray()
    georef = ds.GetGeoTransform()
    del ds

    return (data, georef[0], georef[3], georef[1])

class ImageWriter(object):
    """
    Class that handles writing out image data with GDAL
    """
    def __init__(self, filename, numBands=1, gdalDataType=None,
            driverName=lidarprocessor.DEFAULT_RASTERDRIVERNAME,
            driverOptions=lidarprocessor.DEFAULT_RASTERCREATIONOPTIONS,
            tlx=0.0, tly=0.0, binSize=0.0, epsg=None, nullVal=None,
            ncols=None, nrows=None, calcStats=True):
        """
        Constructor. Set the filename, number of bands etc. If gdalDataType, 
        ncols or nrows is None then these are guessed from the first layer 
        passed to setLayer(). 
        """
        self.ds = None
        self.filename = filename
        self.numBands = numBands
        self.gdalDataType = gdalDataType
        self.driverName = driverName
        self.driverOptions = driverOptions
        self.tlx = tlx
        self.tly = tly
        self.binSize = binSize
        self.epsg = epsg
        self.nullVal = nullVal
        self.ncols = ncols
        self.nrows = nrows
        self.calcStats = calcStats

    def createDataset(self):
        """
        Internal method. Assumes self.gdalDataType, 
        self.ncols and self.nrows is set.
        """
        driver = gdal.GetDriverByName(self.driverName)
        if driver is None:
            msg = 'Unable to find driver for %s' % driverName
            raise SpatialException(msg)

        self.ds = driver.Create(self.filename, self.ncols, self.nrows, 
                    self.numBands, self.gdalDataType, self.driverOptions)
        if self.ds is None:
            msg = 'Unable to create %s' % self.filename
            raise SpatialException(msg)

        self.ds.SetGeoTransform([self.tlx, self.binSize, 0, self.tly, 0, 
                    -self.binSize])

        if self.epsg is not None:
            proj = osr.SpatialReference()
            proj.ImportFromEPSG(self.epsg)
            self.ds.SetProjection(proj.ExportToWkt())

        # Set the null value on every band
        if self.nullVal is not None:
            for i in range(self.numBands):
                band = self.ds.GetRasterBand(i+1)
                band.SetNoDataValue(self.nullVal)

    def setLayer(self, array, layerNum=1):
        """
        Set a layer in the file as a 2d array
        """
        if self.ds is None:
            # create dataset but only get info from array
            # when not given to the constructor.
            if self.gdalDataType is None:
                self.gdalDataType = imageio.NumpyTypeToGDALType(array.dtype)
            if self.ncols is None:
                self.ncols = array.shape[1]
            if self.nrows is None:
                self.nrows = array.shape[0]

            self.createDataset()

        # do some sanity checks to ensure what they pass
        # matches what they have passed/told us before.
        if self.gdalDataType != imageio.NumpyTypeToGDALType(array.dtype):
            msg = 'Data type must be the same for all layers'
            raise SpatialException(msg)
        if self.ncols != array.shape[1]:
            msg = 'X size must be the same for all layers'
            raise SpatialException(msg)
        if self.nrows != array.shape[0]:
            msg = 'Y size must be the same for all layers'
            raise SpatialException(msg)

        band = self.ds.GetRasterBand(layerNum)
        band.WriteArray(array)
        self.ds.FlushCache()

    def close(self):
        """
        Close and flush the dataset, plus calculate stats
        """
        if self.calcStats:
            calcstats.calcStats(self.ds, ignore=self.nullVal)
        self.ds.FlushCache()
        self.ds = None
