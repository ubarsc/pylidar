
"""
Classes that are passed to the user's function
"""

class UserInfo(object):
    # equivalent to rios 'info'
    pass

class DataContainer(object):
    "UserInfo object plus instances of LidarData and ImageData"
    self.info = UserInfo()

class LidarData(object):
    def __init__(self, mode, fileObj):
        self.mode = mode
        self.fileObj = fileObj
        self.extent = None
        
    def setExtent(self, extent):
        self.extent = extent
        
    def getPoints(self):
        "as a structured array"
        points = fileObj.readPointsForExtent(self.extent)
        return points
        
    def getPulses(self):
        "as a structured array"
        pulses = fileObj.readPulsesForExtent(self.extent)
        return pulses
        
    def regridData(self, data):
        "tdb"
        
    def setPoints(self, pts):
        "as a structured array"
        fileObj.writePointsForExtent(self.extent, points)
        
    def setPulses(self, pulses):
        "as a structured array"
        fileObj.writePulsesForExtent(self.extent, pulses)

class ImageData(object):
    def __init__(self, mode):
        self.mode = mode
        
    def getData(self):
        "as 3d array"
        
    def setData(self, data):
        "as 3d array"    
            