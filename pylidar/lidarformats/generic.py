
"""
Base class for LiDAR format reader/writers
"""

READ = 0
UPDATE = 1
CREATE = 2

class Extent(object):
    def __init__(self, tlx=None, brx=None, tly=None, bry=None, binsize=None):
        self.tlx = tlx
        self.brx = brx
        self.tly = tly
        self.bry = bry
        self.binsize = binsize
        

class LiDARFile(object):
    def __init__(self, fname, mode):
        pass
    def readPointsForExtent(self, extent):
        pass
    def readPulsesForExtent(self, extent):
        pass
    def writePointsForExtent(self, extent, data):
        pass
    def writePulsesForExtent(self, extent, data):
        pass
    # see below for no spatial index
    def readPoints(self, n):
        pass
    def readPulses(self, n):
        pass
    def writePoints(self, data):
        pass
    def writePulses(self, data):
        pass



