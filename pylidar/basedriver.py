
"""
Generic 'driver' class. To be subclassed by both 
LiDAR and raster drivers.
"""

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


class Driver(object):
    def __init__(self, fname, mode, controls):
        self.fname = fname 
        self.mode = mode
        self.controls = controls
        
    def setExtent(self, extent):
        raise NotImplementedError()
        
    def getPixelGrid(self):
        raise NotImplementedError()
        
    def setPixelGrid(self, pixGrid):
        raise NotImplementedError()
        
    def close(self):
        raise NotImplementedError()
        
        
