
"""
Driver for GDAL supported files
"""
import numpy
from osgeo import gdal
from rios import pixelgrid
from rios import imageio
from rios.imagereader import ImageReader
from .lidarformats import generic
from . import basedriver

class GDALException(Exception):
    pass

class GDALDriver(basedriver.Driver):
    def __init__(self, fname, mode, controls):
        basedriver.Driver.__init__(self, fname, mode, controls)
        
        if mode != generic.CREATE:
    
            if mode == generic.READ:
                gdalMode = gdal.GA_ReadOnly
            elif mode == generic.UPDATE:
                gdalMode = gdal.GA_Update
    
            self.ds = gdal.Open(fname, gdalMode)
            
            # get the nodata values for readBlockWithMargin
            self.nullValList = []
            for band in range(self.ds.RasterCount):
                bh = self.ds.GetRasterBand(band+1)
                ignore = bh.GetNoDataValue()
                self.nullValList.append(ignore)
            
            self.geoTrans = self.ds.GetGeoTransform()
            self.gdalType = self.ds.GetRasterBand(1).DataType
            
            self.pixGrid = None # unused in read case
            
        else:
            # can't do anything actually until the pixel grid is set
            self.ds = None
            self.gdalType = None
            
            self.pixGrid = None # set by setPixelGrid
            self.geoTrans = None # set by setPixelGrid
            self.nullValList = None # not used in write case
            
        # to read/write at. Set by setExtent()
        self.blockxcoord = None
        self.blockycoord = None
        self.blockxsize = None
        self.blockysize = None
                
    def setExtent(self, extent):
    
        self.blockxcoord, self.blockycoord = gdal.ApplyGeoTransform(self.invGeoTrans,
                                    extent.xMin, extent.yMax)
        self.blockxcoord = int(self.blockxcoord)                            
        self.blockycoord = int(self.blockycoord)                            
                                    
        self.blockxsize = int(numpy.ceil((extent.xMax - extent.xMin) / extent.binSize))
        self.blockysize = int(numpy.ceil((extent.yMax - extent.yMin) / extent.binSize))
                    
    def getPixelGrid(self):
        pixGrid = pixelgrid.pixelGridFromFile(self.fname)
        return pixGrid
                                        
    def setPixelGrid(self, pixGrid):
        # so we can use it in setData to create
        # the dataset when we know the type, bands etc
        self.pixGrid = pixGrid
        self.geoTrans = self.pixGrid.makeGeoTransform()
        success, self.invGeoTrans = gdal.InvGeoTransform(self.geoTrans)
            
    def close(self):
        from rios import calcstats
        calcstats.calcStats(self.ds, self.controls.rasterIgnore)
        self.ds.FlushCache()
        self.ds = None

    def getData(self):
    
        numpyType = imageio.GDALTypeToNumpyType(self.gdalType)
        data = ImageReader.readBlockWithMargin(self.ds, self.blockxcoord, 
                        self.blockycoord, self.blockxsize, self.blockysize, numpyType,
                        self.controls.overlap, self.nullValList)
        return data
        
    def setData(self, data):
        if data.ndim != 3:
            msg = 'Only 3d arrays can be written'
            raise GDALException(msg)

        assert(data.shape[-1] == self.blockxsize)
        assert(data.shape[-2] == self.blockysize)
            
        if self.ds is None:
            # need to create output file first
            nbands = data.shape[0]
            self.gdalType = imageio.NumpyTypeToGDALType(data.dtype)
    
            projection = self.pixGrid.projection
            nrows, ncols = self.pixGrid.getDimensions()
        
            driverName = self.controls.rasterDriver
            driver = gdal.GetDriverByName(driverName)
            if driver is None:
                msg = 'Unable to find driver for %s' % driverName
                raise GDALException(msg)
            
            self.ds = driver.Create(self.fname, ncols, nrows, nbands, 
                            self.gdalType)
            if self.ds is None:
                msg = 'Unable to create %s' % self.fname
                raise GDALException(msg)
                
            self.ds.SetGeoTransform(self.geoTrans)
            self.ds.SetProjection(projection)
            self.nullValList = []
            for band in range(self.ds.RasterCount):
                bh = self.ds.GetRasterBand(band+1)
                bh.SetNoDataValue(self.controls.rasterIgnore)
                self.nullValList.append(self.controls.rasterIgnore)
        
        for band in range(self.ds.RasterCount):
        
            bh = self.ds.GetRasterBand(band + 1)
            # take off overlap if present
            overlap = self.controls.overlap
            slice_bottomMost = data.shape[-2] - overlap
            slice_rightMost = data.shape[-1] - overlap
                                                        
            outblock = data[band, overlap:slice_bottomMost, overlap:slice_rightMost]
                                                                                                
            bh.WriteArray(outblock, self.blockxcoord, self.blockycoord)
            