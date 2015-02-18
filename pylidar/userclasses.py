
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
    # TODO: caching
    def __init__(self, mode):
        self.mode = mode
        
    def getPoints(self):
        "as a structured array"
        
    def getPulses(self):
        "as a structured array"
        
    def regridData(self, data):
        "tdb"
        
    def setPoints(self, pts):
        "as a structured array"
        
    def setPulses(self, pulses):
        "as a structured array"
        
class ImageData(object):
    def __init__(self, mode):
        self.mode = mode
        
    def getData(self):
        "as 3d array"
        
    def setData(self, data):
        "as 3d array"    
            