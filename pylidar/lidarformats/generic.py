
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
    
class LiDARInvalidData(LiDARFileException):
    "Something is wrong with the data read or given"
    
class LiDARInvalidSetting(LiDARFileException):
    "Setting does not make sense"

class LiDARFile(basedriver.Driver):
    """
    Base class for all LiDAR Format reader/writers
    """
    def __init__(self, fname, mode, controls, userClass):
        basedriver.Driver.__init__(self, fname, mode, controls, userClass)
        
    def readPointsForExtent(self):
        """
        Read all the points within the given extent
        as 1d strcutured array
        """
        raise NotImplementedError()
        
    def readPulsesForExtent(self):
        """
        Read all the pulses within the given extent
        as 1d strcutured array
        """
        raise NotImplementedError()
        
    def readTransmitted(self, pulses):
        """
        Read the transmitted waveform for the given (1d) array of pulses
        returns a 2d masked array
        """
        raise NotImplementedError()
        
    def readReceived(self, pulse):
        """
        Read the received waveform for the given (1d) array of pulses
        returns a 2d masked array
        """
        raise NotImplementedError()
        
    def writePointsForExtent(self, points):
        raise NotImplementedError()
        
    def writePulsesForExtent(self, pulses):
        raise NotImplementedError()
        
    def writeTransmitted(self, pulses, transmitted):
        raise NotImplementedError()
        
    def writeReceived(self, pulses, received):
        raise NotImplementedError()
        
    def hasSpatialIndex(self):
        """
        Returns True if file has a spatial index defined
        """
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


def getReaderForLiDARFile(fname, mode, controls, userClass):
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
            inst = cls(fname, mode, controls, userClass)
            # worked - return it
            return inst
        except LiDARFileException:
            # failed - onto the next one
            pass
    # none worked
    msg = 'Cannot open LiDAR file %s' % fname
    raise LiDARFormatDriverNotFound(msg)

