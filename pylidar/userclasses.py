
"""
Classes that are passed to the user's function
"""
import copy

class UserInfo(object):
    # equivalent to rios 'info'
    def __init__(self):
        self.pixGrid = None
        self.extent = None
        
    def setPixGrid(self, pixGrid):
        # take a copy so the user can't change it
        self.pixGrid = copy.copy(pixGrid)
        
    def setExtent(self, extent):
        # take a copy so the user can't change it
        self.extent = copy.copy(extent)

class DataContainer(object):
    "UserInfo object plus instances of LidarData and ImageData"
    def __init__(self):
        self.info = UserInfo()

class LidarData(object):
    def __init__(self, mode, driver):
        self.mode = mode
        self.driver = driver
        self.extent = None
        
    def getPoints(self):
        "as a structured array"
        points = self.driver.readPointsForExtent()
        return points
        
    def getPulses(self):
        "as a structured array"
        pulses = self.driver.readPulsesForExtent()
        return pulses
        
    def regridData(self, data):
        "tdb"
        
    def setPoints(self, pts):
        "as a structured array"
        self.driver.writePointsForExtent(points)
        
    def setPulses(self, pulses):
        "as a structured array"
        self.driver.writePulsesForExtent(pulses)
        
class ImageData(object):
    def __init__(self, mode, driver):
        self.mode = mode
        self.driver = driver
        
    def getData(self):
        "as 3d array"
        self.driver.getData()
        
    def setData(self, data):
        "as 3d array"    
        self.driver.setData(data)
            
            