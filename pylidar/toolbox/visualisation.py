"""
Functions which can be used to help with the visualisation of the point clouds
"""
# This file is part of PyLidar
# Copyright (C) 2015 John Armston, Pete Bunting, Neil Flood, Sam Gillingham
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import print_function, division

import os
import numpy

havepylidarviewer = True
try:
    import pylidarviewer
except ImportError as pylidarviewerErr:
    havepylidarviewer = False
    
haveMatPlotLib = True
try:
    import matplotlib.colors as clr
    import matplotlib.cm as cm
except ImportError as pltErr:
    haveMatPlotLib = False

class VisualisationError(Exception):
    "A Visualisation error has occured"

def rescaleRGB(r, g, b, bit8=True):
    """
    A function which rescales the RGB arrays to a range of 0-1. Where 
    bit8 is True the arrays will just be divided by 255 otherwise a
    min-max rescaling will be applied independently to each array.
    """
    
    if bit8:
        r = r/255
        g = g/255
        b = b/255
    else:
        rMin = numpy.min(r)
        rMax = numpy.max(r)
        r = (r-rMin)/(rMax-rMin)
        
        gMin = numpy.min(g)
        gMax = numpy.max(g)
        g = (g-gMin)/(gMax-gMin)
        
        bMin = numpy.min(b)
        bMax = numpy.max(b)
        b = (b-bMin)/(bMax-bMin)
    return r,g,b

def getClassColoursDict():
    colourDict = {}
    
    colourDict[0] = [1,1,1,1] # Unknown
    colourDict[1] = [0,0,0,1] # Unclassified
    colourDict[2] = [0,0,0,1] # Created
    colourDict[3] = [1,0,0,2] # Ground
    colourDict[4] = [0,1,0,1] # Low Veg
    colourDict[5] = [0,0.803921569,0,1] # Med Veg
    colourDict[6] = [0,0.501960784,0,1] # High Veg
    colourDict[7] = [0.545098039,0.352941176,0,1] # Building
    colourDict[8] = [0,0,1,1] # Water
    colourDict[9] = [0.545098039,0.270588235,0.003921569,1] # Trunk
    colourDict[10] = [0,0.803921569,0,1] # Foliage
    colourDict[11] = [0.803921569,0.2,0.2,1] # Branch
    colourDict[12] = [0.721568627,0.721568627,0.721568627,1] # Wall
    
    return colourDict

def colourByClass(classArr, colourDict=None):
    """
    A function which returns RGB and point size arrays given an input
    array of numerical classes. Where colourDict has been specified then 
    the default colours (specified in getClassColoursDict()) can be overiden.
    """
    if colourDict == None:
        colourDict = getClassColoursDict()
    
    classPres = numpy.unique(classArr)
    
    r = numpy.zeros_like(classArr, dtype=numpy.float)
    g = numpy.zeros_like(classArr, dtype=numpy.float)
    b = numpy.zeros_like(classArr, dtype=numpy.float)
    s = numpy.ones_like(classArr, dtype=numpy.float)
    
    
    for classVal in classPres:
        r[classArr==classVal] = colourDict[classVal][0]
        g[classArr==classVal] = colourDict[classVal][1]
        b[classArr==classVal] = colourDict[classVal][2]
        s[classArr==classVal] = colourDict[classVal][3]
    
    return r,g,b,s


def createRGB4Param(data, stretch='linear', colourMap='Spectral'):
    """
    A function to take a data column (e.g., intensity) and colour into a 
    set of rgb arrays for visualisation. colourMap is a matplotlib colour map
    (see http://wiki.scipy.org/Cookbook/Matplotlib/Show_colormaps) for colouring
    the input data. stretch options are 'linear' or 'stddev'
    """
    # Check matplotlib is available
    if not haveMatPlotLib:
        msg = "The matplotlib module is required for this function could not be imported\n\t" + str(pltErr)
        raise VisualisationError(msg)
    
    if stretch == 'stddev':
        min = numpy.min(data)
        max = numpy.max(data)
        stddev = numpy.std(data)
        mean = numpy.mean(data)
        minData = mean - (2 * mean)
        if minData < min:
            minData = min
        maxData = mean + (2 * mean)
        if maxData > max:
            maxData = max
        
        nData = numpy.zeros_like(data, dtype=numpy.float)
        nData = (data - minData)/(maxData - minData)
        nData[nData < 0] = 0
        nData[nData > 1] = 1
        
    else:
        norm = clr.Normalize(numpy.min(data), numpy.max(data)) 
        nData = norm(data)
        
    my_cmap = cm.get_cmap(colourMap)
    rgba = my_cmap(nData)
    
    r = rgba[:,0:1].flatten()
    g = rgba[:,1:2].flatten()
    b = rgba[:,2:3].flatten()
    
    return r,g,b
    
def displayPointCloud(x, y, z, r, g, b, s):
    """
    Display the point cloud in 3D where arrays (all of the same length) are
    provided for the X, Y, Z position of the points and then the RGB (range 0-1)
    values to colour the points and then the point sizes (s).
    
    X and Y values are centred around 0,0 and then X, Y, Z values are rescale 
    before display
    """
    # Check if pylidarviewer is available
    if not havepylidarviewer:
        msg = "The pylidarviewer module is required for this function could not be imported\n\t" + str(pylidarviewerErr)
        raise VisualisationError(msg)
    
    x = x - numpy.min(x)
    y = y - numpy.min(y)
    z = z - numpy.min(z)

    medianX = numpy.median(x)
    medianY = numpy.median(y)

    maxX = numpy.max(x)
    maxY = numpy.max(y)
    maxZ = numpy.max(z)

    maxVal = maxX
    if maxY > maxVal:
        maxVal = maxY
    if maxZ > maxVal:
        maxVal = maxZ

    x = (x-medianX)/maxVal
    y = (y-medianY)/maxVal
    z = z/maxVal
    
    pylidarviewer.DisplayPointCloud(x, y, z, r, g, b, s)

