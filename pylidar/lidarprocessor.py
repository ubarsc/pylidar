
"""
Classes that are passed to the doProcessing function.
And the doProcessing function itself
"""

from rios import imageio
from rios import pixelgrid
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

DEFAULT_WINDOW_SIZE = 100 # metres
DEFAULT_RASTER_DRIVER = 'KEA'

# inputs to the doProcessing

class DataFiles(object):
    pass
    
class OtherArgs(object):
    pass
    
class Controls(object):
    # stuff to come
    def __init__(self):
        self.footprint = INTERSECTION
        self.windowSize = DEFAULT_WINDOW_SIZE
        self.overlap = 0
        self.rasterDriver = DEFAULT_RASTER_DRIVER
        self.rasterIgnore = 0
        
    def setFootprint(self, footprint):
        self.footprint = footprint
        
    def setWindowSize(self, size):
        "in metres"
        self.windowSize = size
        
    def setOverlap(self, overlap):
        "in bins"
        self.overlap = overlap

    # TODO: raster driver and ignor to be per file        
    def setRasterDriver(self, driverName):
        self.RasterDriver = driverName
        
    def setRasterIgnore(self, ignore):
        self.rasterIgnore = ignore
    
class LidarFile(object):
    def __init__(self, fname, mode):
        # TODO: extra driver options passed as GDAL style list of strings??
        self.fname = fname
        self.mode = mode
    
class ImageFile(object):
    def __init__(self, fname, mode):
        # TODO: extra driver options passed as GDAL style list of strings??
        self.fname = fname
        self.mode = mode
    
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
                                inputFile.mode, controls)
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
                                inputFile.mode, controls)
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


    # close all the files
    for driver in driverList:
        driver.close()

    # 1. Get Extent. Either from files for the controls
    # 2. Read the spatial indices (if being used)
    # 3. Work out where the first block is
    # 4. Loop through each block doing:
    #   4a Read data and assemble objects for user function
    #   4b call user function
    #   4c Write any output data
    #   4d update output spatial index
        


