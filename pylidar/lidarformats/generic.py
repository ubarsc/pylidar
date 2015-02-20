
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
                    binSize=None):
        self.xMin = xMin
        self.xMax = xMax
        self.yMin = yMin
        self.yMax = yMax
        self.binSize = binSize
        
    def __eq__(self, other):
        return (self.xMin == other.xMin and self.xMax == other.xMax and
            self.yMin == other.yMin and self.yMax == other.yMax and
            self.binSize == other.binSize)
            
    def __ne__(self, other):
        return (self.xMin != other.xMin or self.xMax != other.xMax or
            self.yMin != other.yMin or self.yMax != other.yMax or
            self.binSize != other.binSize)
        
    def __str__(self):
        s = "xMin:%s,xMax:%s,yMin:%s,yMax:%s,binSize:%s" % (self.xMin, self.xMax,
                      self.yMin, self.yMax, self.binSize)
        return s

class LiDARFile(object):
    """
    Base class for all LiDAR Format reader/writers
    """
    def __init__(self, fname, mode):
        pass
        
    def getPixelGrid(self):
        raise NotImplementedError()
        
    def setPixelGrid(self, pixGrid):
        raise NotImplementedError()
        
    def readPointsForExtent(self, extent):
        raise NotImplementedError()
        
    def readPulsesForExtent(self, extent):
        raise NotImplementedError()
        
    def writePointsForExtent(self, extent, points):
        raise NotImplementedError()
        
    def writePulsesForExtent(self, extent, pulses):
        raise NotImplementedError()
        
    # see below for no spatial index
    def readPoints(self, n):
        raise NotImplementedError()
        
    def readPulses(self, n):
        raise NotImplementedError()
        
    def writePoints(self, points):
        raise NotImplementedError()
        
    def writePulses(self, pulses):
        raise NotImplementedError()
        
    def close(self, headerInfo=None):
        raise NotImplementedError()


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

