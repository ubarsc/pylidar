
"""
Base class for LiDAR format reader/writers
"""

READ = 0
UPDATE = 1
CREATE = 2

class LiDARFileException(Exception):
    "Base class for LiDAR format reader/writers"
    
class LiDARFormatNotUnderstood(LiDARFileException):
    "Raised when driver cannot open file"
    
class LiDARFormatDriverNotFound(LiDARFileException):
    "None of the drivers can open the file"

class Extent(object):
    """
    Class that defines an extent of an area to read or write
    """
    def __init__(self, xMin=None, xMax=None, yMin=None, yMax=None, 
                    binsize=None):
        self.xMin = xMin
        self.xmax = xMax
        self.yMin = yMin
        self.yMax = yMax
        self.binsize = binsize
        
    def __eq__(self, other):
        return (self.xMin == other.xMin and self.xMax == other.xMax and
            self.yMin == other.yMin and self.yMax == other.yMax and
            self.binSize == other.binSize)
            
    def __ne__(self, other):
        return (self.xMin != other.xMin or self.xMax != other.xMax or
            self.yMin != other.yMin or self.yMax != other.yMax or
            self.binSize != other.binSize)
        


class LiDARFile(object):
    """
    Base class for all LiDAR Format reader/writers
    """
    def __init__(self, fname, mode):
        pass
        
    def getPixelGrid(self):
        pass
    def setPixelGrid(self, pixGrid):
        pass
        
    def readPointsForExtent(self, extent):
        pass
    def readPulsesForExtent(self, extent):
        pass
    def writePointsForExtent(self, extent, points):
        pass
    def writePulsesForExtent(self, extent, pulses):
        pass
    # see below for no spatial index
    def readPoints(self, n):
        pass
    def readPulses(self, n):
        pass
    def writePoints(self, points):
        pass
    def writePulses(self, pulses):
        pass
        
    def close(self, headerInfo=None):
        pass


def getReaderForLiDARFile(fname, mode):
    """
    Returns an instance of a LiDAR format
    reader/writer or raises an exception if none
    found for the file.
    """
    # try each subclass
    for cls in LiDARFile.__subclasses__():
        print('trying', cls)
        try:
            # attempt to create it
            inst = cls(fname, mode)
            # worked - return it
            return inst
        except LiDARFileException:
            # failed - onto the next one
            pass
    # none worked
    msg = 'Cannot open LiDAR file %s' % fname
    raise LiDARFormatDriverNotFound(msg)

