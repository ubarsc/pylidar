
"""
Classes that are passed to the doProcessing function.
And the doProcessing function itself
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
from rios import imageio
from rios import pixelgrid
from rios import cuiprogress
from . import basedriver
from . import gdaldriver
from .lidarformats import generic
# import modules implementing subclasses here so 
# we can use the __subclasses__() python feature
from .lidarformats import spdv3
from .lidarformats import spdv4
from .lidarformats import ascii
from .lidarformats import lvisbin
from .lidarformats import lvishdf5

HAVE_FMT_ASCII_ZLIB = ascii.HAVE_ZLIB
HAVE_FMT_RIEGL = True
try:
    from .lidarformats import riegl
except ImportError:
    # libraries not available
    HAVE_FMT_RIEGL = False

HAVE_FMT_LAS = True
try:
    from .lidarformats import las
except ImportError:
    # library not available
    HAVE_FMT_LAS = False

HAVE_FMT_PULSEWAVES = True
try:
    from .lidarformats import pulsewaves
except ImportError:
    # library not available
    HAVE_FMT_PULSEWAVES = False
    
from . import userclasses

READ = generic.READ
"to be passed to ImageData and LidarData class constructors"
UPDATE = generic.UPDATE
"to be passed to ImageData and LidarData class constructors"
CREATE = generic.CREATE
"to be passed to ImageData and LidarData class constructors"

INTERSECTION = imageio.INTERSECTION
"to be passed to Controls.setFootprint()"
UNION = imageio.UNION
"to be passed to Controls.setFootprint()"
BOUNDS_FROM_REFERENCE = imageio.BOUNDS_FROM_REFERENCE
"to be passed to Controls.setFootprint()"

MESSAGE_WARNING = generic.MESSAGE_WARNING
"""
to be passed to message handler function set with
Controls.setMessageHandler
"""
MESSAGE_INFORMATION = generic.MESSAGE_INFORMATION
"""
to be passed to message handler function set with
Controls.setMessageHandler
"""
MESSAGE_DEBUG = generic.MESSAGE_DEBUG
"""
to be passed to message handler function set with
Controls.setMessageHandler
"""

DEFAULT_WINDOW_SIZE = 256 # bins
"Size of the default window size in bins"

ARRAY_TYPE_POINTS = generic.ARRAY_TYPE_POINTS
"""
For use in userclass.LidarData.translateFieldNames() and 
LiDARFile.getTranslationDict()
"""
ARRAY_TYPE_PULSES = generic.ARRAY_TYPE_PULSES
"""
For use in userclass.LidarData.translateFieldNames() and 
LiDARFile.getTranslationDict()
"""
ARRAY_TYPE_WAVEFORMS = generic.ARRAY_TYPE_WAVEFORMS
"""
For use in userclass.LidarData.translateFieldNames() and 
LiDARFile.getTranslationDict()
"""

CLASSIFICATION_CREATED = generic.CLASSIFICATION_CREATED
"""
Classification codes from the LAS spec. Drivers perform
automatic translation to/from their internal codes for
recognised values.
"""
CLASSIFICATION_UNCLASSIFIED = generic.CLASSIFICATION_UNCLASSIFIED
"Classification codes from the LAS spec."
CLASSIFICATION_GROUND = generic.CLASSIFICATION_GROUND
"Classification codes from the LAS spec."
CLASSIFICATION_LOWVEGE = generic.CLASSIFICATION_LOWVEGE
"Classification codes from the LAS spec."
CLASSIFICATION_MEDVEGE = generic.CLASSIFICATION_MEDVEGE
"Classification codes from the LAS spec."
CLASSIFICATION_HIGHVEGE = generic.CLASSIFICATION_HIGHVEGE
"Classification codes from the LAS spec."
CLASSIFICATION_BUILDING = generic.CLASSIFICATION_BUILDING
"Classification codes from the LAS spec."
CLASSIFICATION_LOWPOINT = generic.CLASSIFICATION_LOWPOINT
"Classification codes from the LAS spec."
CLASSIFICATION_HIGHPOINT = generic.CLASSIFICATION_HIGHPOINT
"Classification codes from the LAS spec."
CLASSIFICATION_WATER = generic.CLASSIFICATION_WATER
"Classification codes from the LAS spec."
CLASSIFICATION_RAIL = generic.CLASSIFICATION_RAIL
"Classification codes from the LAS spec."
CLASSIFICATION_ROAD = generic.CLASSIFICATION_ROAD
"Classification codes from the LAS spec."
CLASSIFICATION_BRIDGE = generic.CLASSIFICATION_BRIDGE
"Classification codes from the LAS spec."
CLASSIFICATION_WIREGUARD = generic.CLASSIFICATION_WIREGUARD
"Classification codes from the LAS spec."
CLASSIFICATION_WIRECOND = generic.CLASSIFICATION_WIRECOND
"Classification codes from the LAS spec."
CLASSIFICATION_TRANSTOWER = generic.CLASSIFICATION_TRANSTOWER
"Classification codes from the LAS spec."
CLASSIFICATION_INSULATOR = generic.CLASSIFICATION_INSULATOR
"Classification codes from the LAS spec."
CLASSIFICATION_TRUNK = generic.CLASSIFICATION_TRUNK
"Extended classification codes"
CLASSIFICATION_FOLIAGE = generic.CLASSIFICATION_FOLIAGE
"Extended classification codes"
CLASSIFICATION_BRANCH = generic.CLASSIFICATION_BRANCH
"Extended classification codes"

def setDefaultDrivers():
    """
    Adapted from RIOS
    Sets some default values into global variables, defining
    what defaults we should use for GDAL and LiDAR  drivers. On any given
    output file these can be over-ridden, and can be over-ridden globally
    using the environment variables (for GDAL):

    * $PYLIDAR_DFLT_RASTERDRIVER
    * $PYLIDAR_DFLT_RASTERDRIVEROPTIONS

    (And for LiDAR):

    * $PYLIDAR_DFLT_LIDARDRIVER
    
    If PYLIDAR_DFLT_RASTERDRIVER is set, then it should be a gdal short driver name
    If PYLIDAR_DFLT_RASTERDRIVEROPTIONS is set, it should be a space-separated list
    of driver creation options, e.g. "COMPRESS=LZW TILED=YES", and should
    be appropriate for the selected GDAL driver. This can also be 'None'
    in which case an empty list of creation options is passed to the driver.
    
    If not otherwise supplied, the default is to use what RIOS is set to. 
    This defaults to the HFA driver with compression. 
        
    If PYLIDAR_DFLT_LIDARDRIVER is set, then is should be a LiDAR driver name
    If not otherwise supplied, the default is to use the SPDV4 driver.
    
    """
    global DEFAULT_RASTERDRIVERNAME, DEFAULT_RASTERCREATIONOPTIONS
    global DEFAULT_LIDARDRIVERNAME
    DEFAULT_RASTERDRIVERNAME = os.getenv('PYLIDAR_DFLT_RASTERDRIVER')
    if DEFAULT_RASTERDRIVERNAME is None:
        # get from rios
        from rios import applier
        DEFAULT_RASTERDRIVERNAME = applier.DEFAULTDRIVERNAME
        
    creationOptionsStr = os.getenv('PYLIDAR_DFLT_RASTERDRIVEROPTIONS')
    if creationOptionsStr is not None:
        if creationOptionsStr == 'None':
            # hack for KEA which needs no creation options
            # and LoadLeveler which deletes any env variables
            # set to an empty values
            DEFAULT_RASTERCREATIONOPTIONS = []
        else:
            DEFAULT_RASTERCREATIONOPTIONS = creationOptionsStr.split()
    else:
        # get from rios
        from rios import applier
        DEFAULT_RASTERCREATIONOPTIONS = applier.DEFAULTCREATIONOPTIONS
            
    DEFAULT_LIDARDRIVERNAME = os.getenv('PYLIDAR_DFLT_LIDARDRIVER', default='SPDV4')
    # Leave driver options for now - info seems likely to be too complex to hold
    # in an environment variable. 

setDefaultDrivers()

# inputs to the doProcessing

class DataFiles(object):
    """
    Container class that has all instances of LidarFile and ImageFile
    inserted into it as the names they are to be used inside the users
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
    
def defaultMessageFn(message, level):
    """
    Default message printer. Prints all messages regardless of level.
    
    Change with Controls.setMessageHandler
    """
    print(message)
    
def silentMessageFn(message, level):
    """
    Alternate message printer - does nothing.
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
        self.spatialProcessing = False
        self.referenceImage = None
        self.referencePixgrid = None
        self.referenceResolution = None
        self.snapGrid = False
        self.progress = cuiprogress.SilentProgress()
        self.messageHandler = defaultMessageFn
        
    def setFootprint(self, footprint):
        """
        Set the footprint of the processing area. This should be
        either INTERSECTION, UNION or BOUNDS_FROM_REFERENCE.

        Note: setting spatial processing to True now deprecated. Consider
        updating your code.
        """
        msg = 'Note: spatial processing now deprecated'
        self.messageHandler(msg, MESSAGE_WARNING)
        self.footprint = footprint
        
    def setWindowSize(self, size):
        """
        Size of the window in bins/pixels that the processing is to be
        performed in. Same in the X and Y direction.
        If doing non spatial processing 'size*size' pulses are read in at
        each iteration.
        """
        self.windowSize = size
        
    def setOverlap(self, overlap):
        """
        Sets the overlap between each window. In bins.

        Note: setting spatial processing to True now deprecated. Consider
        updating your code.
        """
        msg = 'Note: spatial processing now deprecated'
        self.messageHandler(msg, MESSAGE_WARNING)
        self.overlap = overlap

    def setSpatialProcessing(self, spatial):
        """
        Set whether to do processing in a spatial manner. If set to True
        and if one of more LiDAR inputs do not support spatial indexing
        will be reset to False and warning printed.

        Note: setting spatial processing to True now deprecated. Consider
        updating your code.
        """
        if spatial:
            msg = 'Note: spatial processing now deprecated'
            self.messageHandler(msg, MESSAGE_WARNING)
        self.spatialProcessing = spatial
        
    def setReferenceImage(self, referenceImage):
        """
        The path to a reference GDAL image to use when the footprint is
        set to BOUNDS_FROM_REFERENCE. Set only one of this or referencePixgrid
        not both.

        Note: setting spatial processing to True now deprecated. Consider
        updating your code.
        """
        msg = 'Note: spatial processing now deprecated'
        self.messageHandler(msg, MESSAGE_WARNING)
        self.referenceImage = referenceImage
        
    def setReferencePixgrid(self, referencePixgrid):
        """
        The instance of rios.pixelgrid.PixelGridDefn to use as a reference
        when footprint is set to BOUNDS_FROM_REFERENCE. Set only one of this
        or referenceImage, not both.

        Note: setting spatial processing to True now deprecated. Consider
        updating your code.
        """
        msg = 'Note: spatial processing now deprecated'
        self.messageHandler(msg, MESSAGE_WARNING)
        self.referencePixgrid = referencePixgrid
        
    def setReferenceResolution(self, resolution):
        """
        Overrides the resolution that the processing happens with. Overrides
        either of the setReferenceImage or setReferencePixgrid calls or the
        default reference.

        Note: setting spatial processing to True now deprecated. Consider
        updating your code.
        """
        msg = 'Note: spatial processing now deprecated'
        self.messageHandler(msg, MESSAGE_WARNING)
        self.referenceResolution = resolution
        
    def setSnapGrid(self, snap):
        """
        Snap the output grid to be multiples of the resolution. This is only
        needed when ReferenceResolution is not set. True or False.

        Note: setting spatial processing to True now deprecated. Consider
        updating your code.
        """
        msg = 'Note: spatial processing now deprecated'
        self.messageHandler(msg, MESSAGE_WARNING)
        self.snapGrid = snap
        
    def setProgress(self, progress):
        """
        Set the progress instance to use. Usually one of rios.cuiprogress.*
        Default is silent progress
        """
        self.progress = progress

    def setMessageHandler(self, messageHandler):
        """
        Set the message handler function to use for printing messages regarding
        things discovered during the processing. The default behaviour is to 
        print all messages. 
        
        Can pass in silentMessageFn which will print nothing, or your own
        function that takes a message string and a level (one of the
        MESSAGE_* constants).
        """
        self.messageHandler = messageHandler        
    
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
        self.lidarDriverOptions = {}
        self.writeSpatialIndex = True
        
    def setLiDARDriver(self, driverName):
        """
        Set the name of the Lidar driver to use for creaton
        """
        if self.mode != CREATE:
            msg = 'Only valid for creation'
            raise generic.LiDARInvalidSetting(msg)
        self.lidarDriver = driverName
        
    def setLiDARDriverOption(self, key, value):
        """
        Set a key and value that the specific driver understands
        """
        self.lidarDriverOptions[key] = value
        
    def setWriteSpatialIndex(self, writeSpatialIndex):
        """
        Set whether to write spatial index or not on creation or update.
        Ignored for reading.
        """
        self.writeSpatialIndex = writeSpatialIndex
        
    
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
        self.rasterDriver = driverName
        
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
        if self.mode == READ:
            msg = 'Only valid for creation or update'
            raise generic.LiDARInvalidSetting(msg)
        self.rasterIgnore = ignore
    
def doProcessing(userFunc, dataFiles, otherArgs=None, controls=None):
    """
    Main function in PyLidar. Calls function userFunc with each block
    of data. dataFiles to be an instance of DataFiles with fields of instances
    of LidarFile and ImageFile. The names of the fields are re-used in the 
    object passed to userFunc that contains the actual data.
    
    If otherArgs (an instance of OtherArgs) is not None, this is passed as
        the second param to userFunc.
        
    If controls (an instance of Controls) is not None then these controls
        are used for changing the behaviour of reading and writing.
    """
    if controls is None:
        # default values
        controls = Controls()

    # object to be passed to the user function
    userContainer = userclasses.DataContainer(controls)

    # First Open all the files
    gridList, driverList = openFiles(dataFiles, userContainer, controls)
            
    # need to determine if we have a spatial index for all LiDAR files
    if controls.spatialProcessing:
        for driver in driverList:
            if driver.mode != generic.CREATE and isinstance(driver, generic.LiDARFile):
                if not driver.hasSpatialIndex():
                    msg = """Warning: Not all LiDAR files have a spatial index. 
Non-spatial processing will now occur. 
To suppress this message call Controls.setSpatialProcessing(False)"""
                    controls.messageHandler(msg, MESSAGE_WARNING)
                    controls.spatialProcessing = False
                    break
                    
    if not controls.spatialProcessing:
        # need to check no image inputs in non spatial mode
        # this is not an else to the above if since we may
        # have just reset the mode above
        for driver in driverList:
            if isinstance(driver, gdaldriver.GDALDriver):
                msg = 'Can only process image inputs when doing spatial processing'
                raise generic.LiDARFileException(msg)


    # set up depending on if spatial or non spatial processing
    if controls.spatialProcessing:
        
        workingPixGrid = getWorkingPixGrid(controls, userContainer, 
                                gridList, driverList)
            
        # work out where the first block is
        # controls.windowSize is in bins. Convert to meters
        windowSizeWorld = controls.windowSize * workingPixGrid.xRes
        currentExtent = basedriver.Extent(workingPixGrid.xMin, 
                        workingPixGrid.xMin + windowSizeWorld,
                        workingPixGrid.yMax - windowSizeWorld,
                        workingPixGrid.yMax, workingPixGrid.xRes)
                        
        # handle the file being smaller than the block size
        if currentExtent.xMax > workingPixGrid.xMax:
            currentExtent.xMax = workingPixGrid.xMax
        if currentExtent.yMin < workingPixGrid.yMin:
            currentExtent.yMin = workingPixGrid.yMin

        # work out number of pixels of workingPixGrid - allow 
        # rounding error of up to half a pixel by using round
        xsize = numpy.round((workingPixGrid.xMax - workingPixGrid.xMin) / 
                        workingPixGrid.xRes)
        ysize = numpy.round((workingPixGrid.yMax - workingPixGrid.yMin) /
                        workingPixGrid.yRes)

        # now work out total blocks - ceil() allows for partial blocks
        xtotalblocks = int(numpy.ceil(xsize / controls.windowSize))
        ytotalblocks = int(numpy.ceil(ysize / controls.windowSize))
        nTotalBlocks = xtotalblocks * ytotalblocks
        bMoreToDo = currentExtent.yMax > workingPixGrid.yMin
        
    else:
        windowSizeSq = controls.windowSize * controls.windowSize
        try:
            nTotalPulses = max([driver.getTotalNumberPulses() 
                        if driver.mode != generic.CREATE else -1
                        for driver in driverList])
        except generic.LiDARFunctionUnsupported:
            # handle the fact that some drivers might not know
            # how many pulses they have in total
            nTotalBlocks = -1
        else:
            nTotalBlocks = int(numpy.ceil(nTotalPulses / windowSizeSq))
            
        currentRange = generic.PulseRange(0, windowSizeSq)
        if nTotalBlocks != -1:
            bMoreToDo = currentRange.startPulse < nTotalPulses
        else:
            bMoreToDo = True
            
    nBlocksSoFar = 0
    if nTotalBlocks != -1:
        controls.progress.setProgress(0)

    # loop while we haven't fallen off the bottom of the pixelgrid region
    while bMoreToDo:
        # update the driver classes with the new extent
        if controls.spatialProcessing:
            for driver in driverList:
                driver.setExtent(currentExtent)
            # update info class
            userContainer.info.setExtent(currentExtent)
            # last block yet?
            userContainer.info.lastBlock = nBlocksSoFar == (nTotalBlocks - 1)            
        else:
            bMoreToDo = False # assume we have finished
            for driver in driverList:
                if (driver.mode != generic.CREATE and 
                        driver.setPulseRange(currentRange)):
                    # unless there is actually still more data
                    bMoreToDo = True
            # update info class
            userContainer.info.setRange(currentRange)
            # last block yet? we may not know how many pulses there are
            userContainer.info.lastBlock = not bMoreToDo
        
        # build the function args which is one thing, unless
        # there is user data
        functionArgs = (userContainer,)
        if not otherArgs is None:
            functionArgs += (otherArgs, )
            
        # call it if we still have data
        if bMoreToDo:
            userFunc(*functionArgs)
        
        # no longer first block. Was set to True in UserInfo constructor
        userContainer.info.firstBlock = False
        
        # write anything out that has been queued for output
        if bMoreToDo:
            for name in dataFiles.__dict__.keys():
                userClass = getattr(userContainer, name)
                if isinstance(userClass, list):
                    for userClassItem in userClass:
                        userClassItem.flush()
                else:
                    userClass.flush()

        # we have completed another one - this var is used below
        # for calculating block location
        nBlocksSoFar += 1
        
        if controls.spatialProcessing:
            # update to read in next block
            # try going across first
            xblock = nBlocksSoFar % xtotalblocks
            yblock = nBlocksSoFar // xtotalblocks
            currentExtent.xMin = workingPixGrid.xMin + xblock * windowSizeWorld
            currentExtent.xMax = workingPixGrid.xMin + (xblock+1) * windowSizeWorld
            currentExtent.yMax = workingPixGrid.yMax - yblock * windowSizeWorld
            currentExtent.yMin = workingPixGrid.yMax - (yblock+1) * windowSizeWorld
        
            # partial block
            if currentExtent.xMax > workingPixGrid.xMax:
                currentExtent.xMax = workingPixGrid.xMax
        
            # partial block
            if currentExtent.yMin < workingPixGrid.yMin:
                currentExtent.yMin = workingPixGrid.yMin
            
            # done?
            bMoreToDo = (nBlocksSoFar < nTotalBlocks)
        else:
            currentRange.startPulse += windowSizeSq
            currentRange.endPulse += windowSizeSq
            # done?
            # bMoreToDo is updated when the pulse range is set (above)

        # progress
        if nTotalBlocks != -1:
            percentProgress = int((nBlocksSoFar / nTotalBlocks) * 100)
            controls.progress.setProgress(percentProgress)

    controls.progress.reset()
    
    # close all the files
    for driver in driverList:
        driver.close()

def openFiles(dataFiles, userContainer, controls):
    """
    Open all the files required by doProcessing
    """
    gridList = []
    driverList = []
    nameList = dataFiles.__dict__.keys()
    for name in nameList:
        
        inputFiles = getattr(dataFiles, name)
        # check if we are dealing with a list of inputs
        if isinstance(inputFiles, list):
            setattr(userContainer, name, list())
        else:
            inputFiles = [inputFiles]
        
        for inputFile in inputFiles:
            if isinstance(inputFile, LidarFile):
                if inputFile.mode == generic.CREATE:
                    driver = generic.getWriterForLiDARFormat(inputFile.lidarDriver,
                        inputFile.fname, inputFile.mode, controls, inputFile)
                else:
                    driver = generic.getReaderForLiDARFile(inputFile.fname,
                                    inputFile.mode, controls, inputFile)
                driverList.append(driver)

                # create a class to wrap this for the users function
                userClass = userclasses.LidarData(inputFile.mode, driver)
                if hasattr(userContainer, name):
                    getattr(userContainer, name).append(userClass)
                else:
                    setattr(userContainer, name, userClass)

                # grab the pixel grid while we are at it - if reading
                if inputFile.mode != CREATE and controls.spatialProcessing:
                    pixGrid = driver.getPixelGrid()
                    gridList.append(pixGrid)

            elif isinstance(inputFile, ImageFile):
                driver = gdaldriver.GDALDriver(inputFile.fname,
                                    inputFile.mode, controls, inputFile)
                driverList.append(driver)

                # create a class to wrap this for the users function
                userClass = userclasses.ImageData(inputFile.mode, driver)
                if hasattr(userContainer, name):
                    getattr(userContainer, name).append(userClass)
                else:
                    setattr(userContainer, name, userClass)

                # grab the pixel grid while we are at it - if reading
                if inputFile.mode != CREATE:
                    pixGrid = driver.getPixelGrid()
                    gridList.append(pixGrid)

            else:
                msg = "File type not understood"
                raise generic.LiDARFileException(msg)
                
    if len(driverList) == 0:
        msg = 'No input files selected'
        raise generic.LiDARFileException(msg)

    return gridList, driverList

def getWorkingPixGrid(controls, userContainer, gridList, driverList):
    """
    Calculates the working pixel grid and informs the drivers
    and userContainer.
    """
    # work out the reference pixgrid. This is used when the footprint
    # is BOUNDS_FROM_REFERENCE            
    referenceGrid = controls.referencePixgrid
    if referenceGrid is None and controls.referenceImage is not None:
        referenceGrid = pixelgrid.pixelGridFromFile(controls.referenceImage)
    if referenceGrid is None:
        # default to first image
        referenceGrid = gridList[0]

    if controls.referenceResolution is not None:
        # snap one edge to match the new resolution
        res = controls.referenceResolution
        referenceGrid.xMax = res * numpy.ceil(referenceGrid.xMax / res)
        referenceGrid.yMin = res * numpy.floor(referenceGrid.yMin / res)
        referenceGrid.xRes = res
        referenceGrid.yRes = res
        
    elif controls.snapGrid:
        res = referenceGrid.xRes
        referenceGrid.xMin = res * numpy.floor(referenceGrid.xMin / res)
        referenceGrid.xMax = res * numpy.ceil(referenceGrid.xMax / res)
        referenceGrid.yMin = res * numpy.floor(referenceGrid.yMin / res)
        referenceGrid.yMax = res * numpy.ceil(referenceGrid.yMax / res)

    # Check they all have the same projection
    # the LiDAR files don't need to align since we can recompute the spatial 
    # index on the fly.
    for pixGrid in gridList:
        if pixGrid.projection != '' and referenceGrid.projection != '':
            if not referenceGrid.equalProjection(pixGrid):
                msg = 'Un-aligned datasets not yet supported'
                raise generic.LiDARFileException(msg)
        
    # work out common extent
    workingPixGrid = findCommonPixelGridRegion(gridList, referenceGrid, 
                            controls.footprint)
    # we don't support reprojection of raster datasets yet.
    # use RIOS for that. Need to ensure that any input raster datasets
    # are on the workingPixGrid.
    # we can deal with reprojection of LiDAR datasets so don't worry
    # about them.
    for driver in driverList:
        if (isinstance(driver, gdaldriver.GDALDriver) and 
                            driver.mode != CREATE):
            pixGrid = driver.getPixelGrid()
            if not pixGrid.alignedWith(workingPixGrid):
                msg = """Input image file(s) not aligned with calculated 
grid. Resample input images to match, or set grid explicitly with 
controls.setReferenceImage()"""
                raise generic.LiDARFileException(msg)
                            
    # tell all drivers that are creating files what pixel grid is
    for driver in driverList:
        if driver.mode == CREATE:
            driver.setPixelGrid(workingPixGrid)
            
    # tell info class
    userContainer.info.setPixGrid(workingPixGrid)
    
    return workingPixGrid

def findCommonPixelGridRegion(gridList, refGrid, combine=INTERSECTION):
    """
    Returns a PixelGridDefn for the combination of all the grids 
    in the given gridList. The output grid is in the same coordinate 
    system as the reference grid. 
    
    This is adapted from the original in RIOS. This version does not
    attempt to reproject between coordinate systems. Firstly, because
    many LiDAR files do not seem to have the projection set. Secondly,
    we don't support reprojection anyway - unlike RIOS.
    
    The combine parameter controls whether UNION, INTERSECTION 
    or BOUNDS_FROM_REFERENCE is performed. 
    
    """
    newGrid = refGrid
    if combine != imageio.BOUNDS_FROM_REFERENCE:
        for grid in gridList:
            if not newGrid.alignedWith(grid):
                xMin = grid.snapToGrid(grid.xMin, refGrid.xMin, refGrid.xRes)
                xMax = grid.snapToGrid(grid.xMax, refGrid.xMax, refGrid.xRes)
                yMin = grid.snapToGrid(grid.yMin, refGrid.yMin, refGrid.yRes)
                yMax = grid.snapToGrid(grid.yMax, refGrid.yMax, refGrid.yRes)
                grid = pixelgrid.PixelGridDefn(xMin=xMin, xMax=xMax, yMin=yMin, 
                        yMax=yMax, xRes=refGrid.xRes, yRes=refGrid.yRes, 
                        projection=refGrid.projection)

            if combine == imageio.INTERSECTION:
                newGrid = newGrid.intersection(grid)
            elif combine == imageio.UNION:
                newGrid = newGrid.union(grid)
        
    return newGrid
