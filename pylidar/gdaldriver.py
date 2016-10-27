
"""
Driver for GDAL supported files
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

from distutils.version import LooseVersion

import numpy
from osgeo import gdal
from rios import pixelgrid
from rios import imageio
from rios.imagereader import ImageReader
from .lidarformats import generic
from . import basedriver

class GDALException(generic.LiDARFileException):
    """
    An exception that is raised by this driver
    """
    pass

class GDALDriver(basedriver.Driver):
    """
    This driver supports reading and writing of raster data using GDAL.
    """
    def __init__(self, fname, mode, controls, userClass):
        basedriver.Driver.__init__(self, fname, mode, controls, userClass)
        
        if mode != basedriver.CREATE:
            # file already exists
            if mode == basedriver.READ:
                gdalMode = gdal.GA_ReadOnly
            elif mode == basedriver.UPDATE:
                gdalMode = gdal.GA_Update
    
            # open it in the right mode
            self.ds = gdal.Open(fname, gdalMode)
            
            # get the nodata values for readBlockWithMargin
            self.nullValList = []
            for band in range(self.ds.RasterCount):
                bh = self.ds.GetRasterBand(band+1)
                ignore = bh.GetNoDataValue()
                self.nullValList.append(ignore)
            
            # get this other information while we are here
            self.geoTrans = self.ds.GetGeoTransform()
            gdalVersion = LooseVersion(gdal.__version__)
            if gdalVersion < LooseVersion('2.0.0'):
                success, self.invGeoTrans = gdal.InvGeoTransform(self.geoTrans)
            else:
                self.invGeoTrans = gdal.InvGeoTransform(self.geoTrans)
            self.gdalType = self.ds.GetRasterBand(1).DataType
            
            self.pixGrid = None # unused in read case
            
        else:
            # can't do anything actually until the pixel grid is set
            self.ds = None
            self.gdalType = None
            
            self.pixGrid = None # set by setPixelGrid
            self.geoTrans = None # set by setPixelGrid
            self.invGeoTrans = None # set by setPixelGrid
            self.nullValList = None # not used in write case
            
        # to read/write at. Set by setExtent()
        self.blockxcoord = None
        self.blockycoord = None
        self.blockxsize = None
        self.blockysize = None
                
    def setExtent(self, extent):
        """
        Set the extent for the next read or write. Convert from world coords
        to file coords.
        """
        self.blockxcoord, self.blockycoord = gdal.ApplyGeoTransform(self.invGeoTrans,
                                    float(extent.xMin), float(extent.yMax))
        self.blockxcoord = int(self.blockxcoord)                            
        self.blockycoord = int(self.blockycoord)                            

        # round() ok since points should already be on the grid, nasty 
        # rounding errors propogated with ceil()                                    
        self.blockxsize = int(numpy.round((extent.xMax - extent.xMin) / extent.binSize))
        self.blockysize = int(numpy.round((extent.yMax - extent.yMin) / extent.binSize))
                    
    def getPixelGrid(self):
        """
        Get the pixel grid for this file
        """
        pixGrid = pixelgrid.pixelGridFromFile(self.fname)
        return pixGrid
                                        
    def setPixelGrid(self, pixGrid):
        """
        Set the pixel grid to use for this new file
        """
        # so we can use it in setData to create
        # the dataset when we know the type, bands etc
        self.pixGrid = pixGrid
        self.geoTrans = self.pixGrid.makeGeoTransform()
        gdalVersion = LooseVersion(gdal.__version__)
        if gdalVersion < LooseVersion('2.0.0'):
            success, self.invGeoTrans = gdal.InvGeoTransform(self.geoTrans)
        else:
            self.invGeoTrans = gdal.InvGeoTransform(self.geoTrans)
            
    def close(self):
        """
        Calculate stats etc
        """
        from rios import calcstats
        if self.mode != basedriver.READ:
            progress = self.controls.progress
            ignore = self.userClass.rasterIgnore
            calcstats.calcStats(self.ds, progress, ignore)
            self.ds.FlushCache()
        self.ds = None

    def getData(self):
        """
        Read a 3d numpy array with data for the current extent
        """
        if self.mode == basedriver.CREATE:
            msg = 'Can only read raster data in READ or UPDATE modes'
            raise GDALException(msg)
        
        numpyType = imageio.GDALTypeToNumpyType(self.gdalType)
        # use RIOS to do the hard work
        data = ImageReader.readBlockWithMargin(self.ds, self.blockxcoord, 
                        self.blockycoord, self.blockxsize, self.blockysize, numpyType,
                        self.controls.overlap, self.nullValList)
        return data
        
    def setData(self, data):
        """
        Write a 3d numpy array to the image
        """
        if self.mode == basedriver.READ or data is None:
            # the user class always call this as the processor makes no
            # distinction with read/write when calling flush()
            return
        
        if data.ndim != 3:
            msg = 'Only 3d arrays can be written'
            raise GDALException(msg)

        # remove overlap
        nbands, nrows, ncols = data.shape
        nrows -= (self.controls.overlap * 2)
        ncols -= (self.controls.overlap * 2)

        if (ncols != self.blockxsize or 
                nrows != self.blockysize):
            msg = 'data is incorrect size for writing current block'
            raise GDALException(msg)
            
        if self.ds is None:
            # need to create output file first size know we know datatype etc
            self.gdalType = imageio.NumpyTypeToGDALType(data.dtype)
    
            # get info for whole file needed.
            projection = self.pixGrid.projection
            nrows, ncols = self.pixGrid.getDimensions()
        
            # find driver
            driverName = self.userClass.rasterDriver
            driver = gdal.GetDriverByName(driverName)
            if driver is None:
                msg = 'Unable to find driver for %s' % driverName
                raise GDALException(msg)
            
            # get options and create dataset
            driverOptions = self.userClass.rasterDriverOptions
            self.ds = driver.Create(self.fname, ncols, nrows, nbands, 
                            self.gdalType, driverOptions)
            if self.ds is None:
                msg = 'Unable to create %s' % self.fname
                raise GDALException(msg)
                
            # set info on new dataset
            self.ds.SetGeoTransform(self.geoTrans)
            self.ds.SetProjection(str(projection))
            self.nullValList = []
            ignore = self.userClass.rasterIgnore
            for band in range(self.ds.RasterCount):
                bh = self.ds.GetRasterBand(band+1)
                bh.SetNoDataValue(ignore)
                self.nullValList.append(ignore)
                
        else:
            # check they are consistent about the data type
            if self.gdalType != imageio.NumpyTypeToGDALType(data.dtype):
                msg = 'Output data type must stay the same for each block'
                raise GDALException(msg)
                
            # and bands
            if self.ds.RasterCount != nbands:
                msg = 'Output number of bands must stay the same for each block'
                raise GDALException(msg)
        
        # ok now we can write the data
        for band in range(self.ds.RasterCount):
        
            bh = self.ds.GetRasterBand(band + 1)
            # take off overlap if present
            overlap = self.controls.overlap
            slice_bottomMost = data.shape[-2] - overlap
            slice_rightMost = data.shape[-1] - overlap
                                                        
            outblock = data[band, overlap:slice_bottomMost, overlap:slice_rightMost]
            
            # if a masked array full in the masked out areas with 
            # the bands null value
            if isinstance(outblock, numpy.ma.masked_array):
                outblock = outblock.filled(self.nullValList[band])
                                                                                                
            bh.WriteArray(outblock, self.blockxcoord, self.blockycoord)
            
