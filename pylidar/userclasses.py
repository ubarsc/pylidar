
"""
Classes that are passed to the user's function
"""

class UserInfo(object):
    # equivalent to rios 'info'
    pass

class DataContainer(object):
    "UserInfo object plus instances of LidarData and ImageData"
    def __init__(self):
        self.info = UserInfo()

class LidarData(object):
    def __init__(self, mode, driver):
        self.mode = mode
        self.driver = driver
        self.extent = None
        
    def setExtent(self, extent):
        self.extent = extent
        
    def getPoints(self):
        "as a structured array"
        points = driver.readPointsForExtent(self.extent)
        return points
        
    def getPulses(self):
        "as a structured array"
        pulses = driver.readPulsesForExtent(self.extent)
        return pulses
        
    def regridData(self, data):
        "tdb"
        
    def setPoints(self, pts):
        "as a structured array"
        driver.writePointsForExtent(self.extent, points)
        
    def setPulses(self, pulses):
        "as a structured array"
        driver.writePulsesForExtent(self.extent, pulses)
        
    def close(self):
        "close the driver"
        self.driver.close()

class ImageData(object):
    def __init__(self, mode):
        self.mode = mode
        
    def getData(self):
        "as 3d array"
        
    def setData(self, data):
        "as 3d array"    
            