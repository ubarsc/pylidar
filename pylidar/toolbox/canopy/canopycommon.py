"""
Common functions and classes for the canopy module
There is some temporary duplication of functions from the spatial branch
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
from osgeo import osr
from osgeo import gdal
from rios import imageio
from rios import calcstats
from rios import pixelgrid
from pylidar.lidarformats import generic
from pylidar import lidarprocessor


class CanopyMetricError(Exception):
    "Exception type for canopymetric errors"


def prepareInputFiles(infiles, otherargs, index=None):
    """
    Prepare input files for calculation of canopy metrics
    """    
    dataFiles = lidarprocessor.DataFiles()
    if index is not None:
        dataFiles.inFiles = [lidarprocessor.LidarFile(infiles[index], lidarprocessor.READ)]
    else:
        dataFiles.inFiles = [lidarprocessor.LidarFile(fname, lidarprocessor.READ) for fname in infiles]
    
    otherargs.lidardriver = []
    otherargs.proj = []
    
    nFiles = len(dataFiles.inFiles)
    for i in range(nFiles):        
        info = generic.getLidarFileInfo(dataFiles.inFiles[i].fname)
        if info.getDriverName() == 'riegl':
            if otherargs.externaltransformfn is not None:
                if index is not None:
                    externaltransform = numpy.loadtxt(otherargs.externaltransformfn[index], ndmin=2, delimiter=" ", dtype=numpy.float32) 
                else:
                    externaltransform = numpy.loadtxt(otherargs.externaltransformfn[i], ndmin=2, delimiter=" ", dtype=numpy.float32)
                dataFiles.inFiles[i].setLiDARDriverOption("ROTATION_MATRIX", externaltransform)
            elif "ROTATION_MATRIX" in info.header:
                dataFiles.inFiles[i].setLiDARDriverOption("ROTATION_MATRIX", info.header["ROTATION_MATRIX"])
            else:
                msg = 'Input file %s has no valid pitch/roll/yaw data' % dataFiles.inFiles[i].fname
                raise generic.LiDARInvalidData(msg)

        otherargs.lidardriver.append( info.getDriverName() )
        
        if "SPATIAL_REFERENCE" in info.header.keys():
            if len(info.header["SPATIAL_REFERENCE"]) > 0:
                otherargs.proj.append(info.header["SPATIAL_REFERENCE"])
            else:
                otherargs.proj.append(None)
        else:
            otherargs.proj.append(None)
    
    return dataFiles


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
            tlx=0.0, tly=0.0, binSize=0.0, epsg=None, wkt=None, nullVal=None,
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
        self.wkt = wkt
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
        elif self.wkt is not None:
            self.ds.SetProjection(self.wkt) 

            # Set the null value on every band
        if self.nullVal is not None:
            for i in range(self.numBands):
                band = ds.GetRasterBand(i+1)
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

