
"""
Classes that are passed to the doProcessing function.
And the doProcessing function itself
"""
# This file is part of PyLidar
# Copyright (C) 2015 John Armston, Neil Flood and Sam Gillingham
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

import os
from rios import imageio
from rios import pixelgrid
from rios import cuiprogress
from . import basedriver
from . import gdaldriver
from .lidarformats import generic
from .lidarformats import spdv3
from . import userclasses

READ = generic.READ
UPDATE = generic.UPDATE
CREATE = generic.CREATE

INTERSECTION = imageio.INTERSECTION
UNION = imageio.UNION
BOUNDS_FROM_REFERENCE = imageio.BOUNDS_FROM_REFERENCE

DEFAULT_WINDOW_SIZE = 200 # bins

def setDefaultDrivers():
    """
    Adapted from RIOS
    Sets some default values into global variables, defining
    what defaults we should use for GDAL and LiDAR  drivers. On any given
    output file these can be over-ridden, and can be over-ridden globally
    using the environment variables (for GDAL):
        $PYLIDAR_DFLT_RASTERDRIVER
        $PYLIDAR_DFLT_RASTERDRIVEROPTIONS
    (And for LiDAR):
        $PYLIDAR_DFLT_LIDARDRIVER
        $PYLIDAR_DFLT_LIDARDRIVEROPTIONS
    
    If PYLIDAR_DFLT_RASTERDRIVER is set, then it should be a gdal short driver name
    If PYLIDAR_DFLT_RASTERDRIVEROPTIONS is set, it should be a space-separated list
    of driver creation options, e.g. "COMPRESS=LZW TILED=YES", and should
    be appropriate for the selected GDAL driver. This can also be 'None'
    in which case an empty list of creation options is passed to the driver.
    
    If not otherwise supplied, the default is to use the HFA driver, with compression. 
        
    If PYLIDAR_DFLT_LIDARDRIVER is set, then is should be a LiDAR driver anme
    If PYLIDAR_DFLT_LIDARDRIVEROPTIONS is set it should be a space-separated list
    of driver creation options and should be appropriate for the selected LiDAR
    driver. This can also be 'None' in which case an empty list of creation 
    options is passed to the driver.
        
    If not otherwise supplied, the default is to use the SPDV3 driver.
    """
    global DEFAULT_RASTERDRIVERNAME, DEFAULT_RASTERCREATIONOPTIONS
    global DEFAULT_LIDARDRIVERNAME, DEFAULT_LIDARCREATIONOPTIONS
    DEFAULT_RASTERDRIVERNAME = os.getenv('PYLIDAR_DFLT_RASTERDRIVER', default='HFA')
    DEFAULT_RASTERCREATIONOPTIONS = ['COMPRESSED=TRUE','IGNOREUTM=TRUE']
    creationOptionsStr = os.getenv('PYLIDAR_DFLT_RASTERDRIVEROPTIONS')
    if creationOptionsStr is not None:
        if creationOptionsStr == 'None':
            # hack for KEA which needs no creation options
            # and LoadLeveler which deletes any env variables
            # set to an empty values
            DEFAULT_RASTERCREATIONOPTIONS = []
        else:
            DEFAULT_RASTERCREATIONOPTIONS = creationOptionsStr.split()
            
    DEFAULT_LIDARDRIVERNAME = os.getenv('PYLIDAR_DFLT_LIDARDRIVER', default='SPDV3')
    DEFAULT_LIDARCREATIONOPTIONS = []
    creationOptionsStr = os.getenv('PYLIDAR_DFLT_LIDARDRIVEROPTIONS')
    if creationOptionsStr is not None:
        if creationOptionsStr == 'None':
            DEFAULT_LIDARCREATIONOPTIONS = []
        else:
            DEFAULT_LIDARCREATIONOPTIONS = creationOptionsStr.split()

setDefaultDrivers()

# inputs to the doProcessing

class DataFiles(object):
    """
    Container class that has all instances of LidarFile and ImageFile
    insterted into it as the names they are to be used inside the users
    function.
    """
    pass
    
class OtherArgs(object):
    """
    Container class that has any arbitary information that the user function
    requires. Set in the same form as DataFiles above, but no conversion of the
    contents happens.
    """
    pass
    
class Controls(object):
    """
    The controls object. This is passed to the doProcessing function 
    and contains methods for controling the behaviour of the processing.
    """
    def __init__(self):
        self.footprint = INTERSECTION
        self.windowSize = DEFAULT_WINDOW_SIZE
        self.overlap = 0
        self.spatialProcessing = True 
        self.referenceImage = None
        self.referencePixgrid = None
        self.progress = cuiprogress.SilentProgress()
        
    def setFootprint(self, footprint):
        """
        Set the footprint of the processing area. This should be
        either INTERSECTION, UNION or BOUNDS_FROM_REFERENCE.
        """
        self.footprint = footprint
        
    def setWindowSize(self, size):
        """
        Size of the window in bins/pixels that the processing is to be
        performed in. Same in the X and Y direction.
        If doing non spatial processing 'size' pulses are read in at
        each iteration.
        """
        self.windowSize = size
        
    def setOverlap(self, overlap):
        """
        Sets the overlap between each window. In bins.
        """
        self.overlap = overlap

    def setSpatialProcessing(self, spatial):
        """
        Set whether to do processing in a spatial manner. If set to True
        and if one of more LiDAR inputs do not support spatial indexing
        will be reset to False and warning printed.
        """
        self.spatialProcessing = spatial
        
    def setReferenceImage(self, referenceImage):
        """
        The path to a reference GDAL image to use when the footprint is
        set to BOUNDS_FROM_REFERENCE. Set only one of this or referencePixgrid
        not both.
        """
        self.referenceImage = referenceImage
        
    def setReferencePixgrid(self, referencePixgrid):
        """
        The instance of rios.pixelgrid.PixelGridDefn to use as a reference
        when footprint is set to BOUNDS_FROM_REFERENCE. Set only one of this
        or referenceImage, not both.
        """
        self.referencePixgrid = referencePixgrid
        
    def setProgress(self, progress):
        """
        Set the progress instance to use. Usually one of rios.cuiprogress.*
        Default is silent progress
        """
        self.progress = progress
    
class LidarFile(object):
    """
    Create an instance of this to process a LiDAR file. Set it to a 
    field within your instance of DataFiles.
    The mode is one of: READ, UPDATE or CREATE.
    """
    def __init__(self, fname, mode):
        self.fname = fname
        self.mode = mode
        self.lidarDriver = DEFAULT_LIDARDRIVERNAME
        self.lidarDriverOptions = DEFAULT_LIDARCREATIONOPTIONS
        
    def setLiDARDriver(self, driverName):
        """
        Set the name of the Lidar driver to use for creaton
        """
        if self.mode != CREATE:
            msg = 'Only valid for creation'
            raise generic.LiDARInvalidSetting(msg)
        self.lidarDriver = driverName
        
    def setLiDARDriverOptions(self, options):
        """
        Set a list of strings in driver specific format
        """
        if self.mode != CREATE:
            msg = 'Only valid for creation'
            raise generic.LiDARInvalidSetting(msg)
        self.lidarDriverOptions = options
    
class ImageFile(object):
    def __init__(self, fname, mode):
        self.fname = fname
        self.mode = mode
        self.rasterDriver = DEFAULT_RASTERDRIVERNAME
        self.rasterDriverOptions = DEFAULT_RASTERCREATIONOPTIONS
        self.rasterIgnore = 0

    def setRasterDriver(self, driverName):
        """
        Set GDAL driver short name to use for output format.
        """
        if self.mode != CREATE:
            msg = 'Only valid for creation'
            raise generic.LiDARInvalidSetting(msg)
        self.RasterDriver = driverName
        
    def setRasterDriverOptions(self, options):
        """
        Set a list of strings in driver specific format. See GDAL
        documentation.
        """
        if self.mode != CREATE:
            msg = 'Only valid for creation'
            raise generic.LiDARInvalidSetting(msg)
        self.rasterDriverOptions = options
        
    def setRasterIgnore(self, ignore):
        """
        Set the ignore value for calculating statistics
        """
        # TODO: maybe should be valid for read for area outside
        # footprint?
        if self.mode == READ:
            msg = 'Only valid for creation or update'
            raise generic.LiDARInvalidSetting(msg)
        self.rasterIgnore = ignore
    
def doProcessing(userFunc, dataFiles, otherArgs=None, controls=None):

    # TODO: update so we can handle no spatial index
    # -requested via the controls
    # - or file doesn't have one
    # when no spatial index - read 1000 PULSES per time

    if controls is None:
        # default values
        controls = Controls()

    # object to be passed to the user function
    userContainer = userclasses.DataContainer()

    # First Open all the files
    gridList = []
    driverList = []
    for name in dataFiles.__dict__.keys():
        # TODO: lists, dictionaries etc
        inputFile = getattr(dataFiles, name)
        if isinstance(inputFile, LidarFile):
            driver = generic.getReaderForLiDARFile(inputFile.fname,
                                inputFile.mode, controls, inputFile)
            driverList.append(driver)
                                
            # create a class to wrap this for the users function
            userClass = userclasses.LidarData(inputFile.mode, driver)
            setattr(userContainer, name, userClass)
            
            # grab the pixel grid while we are at it - if reading
            if inputFile.mode != CREATE:
                pixGrid = driver.getPixelGrid()
                gridList.append(pixGrid)
                
        elif isinstance(inputFile, ImageFile):
            driver = gdaldriver.GDALDriver(inputFile.fname,
                                inputFile.mode, controls, inputFile)
            driverList.append(driver)

            # create a class to wrap this for the users function
            userClass = userclasses.ImageData(inputFile.mode, driver)
            setattr(userContainer, name, userClass)
                        
            # grab the pixel grid while we are at it - if reading
            if inputFile.mode != CREATE:
                pixGrid = driver.getPixelGrid()
                gridList.append(pixGrid)
                                                                                
        else:
            msg = "File type not understood"
            raise LiDARFileException(msg)
            
    if len(gridList) == 0:
        msg = 'No input files selected'
        raise LiDARFileException(msg)
        
    # TODO: need to determine if we have a spatial index for all LiDAR files
        
    # for now, check they all align
    firstPixGrid = None
    for pixGrid in gridList:
        if firstPixGrid is None:
            firstPixGrid = pixGrid
        else:
            if not firstPixGrid.alignedWith(pixGrid):
                msg = 'Un-aligned datasets not yet supported'
                raise LiDARFileException(msg)
            
    # work out common extent
    # TODO: user input reference grid
    workingPixGrid = pixelgrid.findCommonRegion(gridList, gridList[0], 
                            controls.footprint)
                            
    # tell all drivers that are creating files what pixel grid is
    for driver in driverList:
        if driver.mode != READ:
            driver.setPixelGrid(workingPixGrid)
            
    # info clas
    userContainer.info.setPixGrid(workingPixGrid)
                            
    # work out where the first block is
    currentExtent = basedriver.Extent(workingPixGrid.xMin, 
                        workingPixGrid.xMin + controls.windowSize,
                        workingPixGrid.yMax - controls.windowSize,
                        workingPixGrid.yMax, workingPixGrid.xRes)
                        
    # loop while we haven't fallen off the bottom of the pixelgrid region
    while currentExtent.yMax > workingPixGrid.yMin:
        # update the driver classes with the new extent
        for driver in driverList:
            driver.setExtent(currentExtent)
            
        # info class
        userContainer.info.setExtent(currentExtent)
            
        # build the function args which is one thing, unless
        # there is user data
        functionArgs = (userContainer,)
        if not otherArgs is None:
            functionArgs += (otherArgs, )
            
        # call it
        userFunc(*functionArgs)
        
        # update to read in next block
        # try going accross first
        currentExtent.xMin += controls.windowSize
        currentExtent.xMax += controls.windowSize
        
        # partial block
        if currentExtent.xMax > workingPixGrid.xMax:
            currentExtent.xMax = workingPixGrid.xMax
        
        if currentExtent.xMin > workingPixGrid.xMax:
            # start next line down
            currentExtent.xMin = workingPixGrid.xMin
            currentExtent.xMax = workingPixGrid.xMin + controls.windowSize
            currentExtent.yMax -= controls.windowSize
            currentExtent.yMin -= controls.windowSize
            
        # partial block
        if currentExtent.yMin < workingPixGrid.yMin:
            currentExtent.yMin = workingPixGrid.yMin

        # TODO: progress

    # close all the files
    for driver in driverList:
        driver.close()

