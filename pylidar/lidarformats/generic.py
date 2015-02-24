
"""
Base class for LiDAR format reader/writers
"""

from .. import basedriver

READ = 0
UPDATE = 1
CREATE = 2

class LiDARFileException(Exception):
    "Base class for LiDAR format reader/writers"
    
class LiDARFormatNotUnderstood(LiDARFileException):
    "Raised when driver cannot open file"
    
class LiDARFormatDriverNotFound(LiDARFileException):
    "None of the drivers can open the file"

class LiDARFile(basedriver.Driver):
    """
    Base class for all LiDAR Format reader/writers
    """
    def __init__(self, fname, mode, controls):
        basedriver.Driver.__init__(self, fname, mode, controls)
        
    def getPixelGrid(self):
        raise NotImplementedError()
        
    def setPixelGrid(self, pixGrid):
        raise NotImplementedError()
        
    def setExtent(self, extent):
        raise NotImplementedError()
        
    def readPointsForExtent(self):
        raise NotImplementedError()
        
    def readPulsesForExtent(self):
        raise NotImplementedError()
        
    def readTransmitted(self, pulse):
        raise NotImplementedError()
        
    def readReceived(self, pulse):
        raise NotImplementedError()
        
    def writePointsForExtent(self, points):
        raise NotImplementedError()
        
    def writePulsesForExtent(self, pulses):
        raise NotImplementedError()
        
    def writeTransmitted(self, pulse, transmitted):
        raise NotImplementedError()
        
    def writeReceived(self, pulse, received):
        raise NotImplementedError()
        

    def hasSpatialIndex(self):
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


def getReaderForLiDARFile(fname, mode, controls):
    """
    Returns an instance of a LiDAR format
    reader/writer or raises an exception if none
    found for the file.
    """
    # try each subclass
    for cls in LiDARFile.__subclasses__():
        #print('trying', cls)
        try:
            # attempt to create it
            inst = cls(fname, mode, controls)
            # worked - return it
            return inst
        except LiDARFileException:
            # failed - onto the next one
            pass
    # none worked
    msg = 'Cannot open LiDAR file %s' % fname
    raise LiDARFormatDriverNotFound(msg)

