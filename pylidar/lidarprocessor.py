
"""
Classes that are passed to the doProcessing function.
And the doProcessing function itself
"""

from rios import imageio
from rios import pixelgrid
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
        
    def setFootprint(self, footprint):
        self.footprint = footprint
        
    def setWindowSize(self, size):
        "in metres"
        self.windowSize = size
    
class LidarFile(object):
    def __init__(self, fname, mode):
        # TODO: extra driver options passed as GDAL style list of strings??
        self.fname = fname
        self.mode = mode
    
class ImageFile(object):
    pass
    
def doProcessing(userFunc, dataFiles, otherArgs=None, controls=None):

    if controls is None:
        # default values
        controls = Controls()

    # object to be passed to the user function
    userContainer = userclasses.DataContainer()

    # First Open all the files
    gridList = []
    for name in dataFiles.__dict__.keys():
        inputFile = getattr(dataFiles, name)
        if isinstance(inputFile, LidarFile):
            driver = generic.getReaderForLiDARFile(inputFile.fname,
                                inputFile.mode)
                                
            # create a class to wrap this for the users function
            userClass = userclasses.LidarData(inputFile.mode, driver)
            setattr(userContainer, name, userClass)
            
            # grab the pixel grid while we are at it
            pixGrid = driver.getPixelGrid()
            gridList.append(pixGrid)
        else:
            msg = "image files not supported yet"
            raise LiDARFileException(msg)
            
            
    # work out common extent
    # TODO: user input reference grid
    # should we allow reprojection at all (pixelgrid allows it)?
    workingPixGrid = pixelgrid.findCommonRegion(gridList, gridList[0], 
                            controls.footprint)
                            
    # work out where the first block is
    currentExtent = generic.Extent(workingPixGrid.xMin, 
                        workingPixGrid.xMin + controls.windowSize,
                        workingPixGrid.yMax - controls.windowSize,
                        workingPixGrid.yMax, workingPixGrid.xRes)
                        
    # loop while we haven't fallen off the bottom of the pixelgrid region
    while currentExtent.yMax > workingPixGrid.yMin:
        # update the user classes with the new extent
        # we get the keys from the input rather than userContainer
        # since userContainer may have other things in it
        for name in dataFiles.__dict__.keys():
            userClass = getattr(userContainer, name)
            userClass.setExtent(currentExtent)
            
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
        if currentExtent.xMin > workingPixGrid.xMax:
            # start next line down
            currentExtent.xMin = workingPixGrid.xMin
            currentExtent.xMax = workingPixGrid.xMin + controls.windowSize
            currentExtent.yMax -= controls.windowSize
            currentExtent.yMin -= controls.windowSize


    # close all the files
    for name in dataFiles.__dict__.keys():
        userClass = getattr(userContainer, name)
        userClass.close()
                

    # 1. Get Extent. Either from files for the controls
    # 2. Read the spatial indices (if being used)
    # 3. Work out where the first block is
    # 4. Loop through each block doing:
    #   4a Read data and assemble objects for user function
    #   4b call user function
    #   4c Write any output data
    #   4d update output spatial index
        


