"""
Functions for calculating vertical plant profiles (Calders et al., 2014)
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

import numpy
import collections
import statsmodels.api as sm
from numba import jit

@jit
def stratifyPointsByZenithHeight(midZenithBins,minimumZenith,maximumZenith,zenithBinSize,pulseZenith,pulsesByPointZenith,
                                 pointHeight,heightBins,heightBinSize,pointCounts,pulseCounts,weights):
    """
    Called by runZenithHeightStratification()
    
    Updates a 2D array of point counts (zenith, height) and 1D array of pulse counts (zenith)
    
    Parameters:
        midZenithBins           1D array of midpoints of zenith angle bins to use for the stratification
        minimumZenith           Minimum zenith angle value to consider for this block
        maximumZenith           Maximum zenith angle value to consider for this block
        zenithBinSize           Zenith angle bin size
        pulseZenith             1D array of pulse zenith angles for this block
        pulsesByPointZenith     1D array of pulse zenith angles for each point in this block
        pointHeight             1D array of point heights for this block
        heightBins              1D array of vertical height bin starts to use the stratification
        heightBinSize           Vertical height bin size
        weights                 1D array of points weights to use for calculating point intercept counts
        
    Returns:    
        pointCounts             2D array (zenith bins, height bins) of point intercept counts to update for this block
        pulseCounts             1D array (zenith bins) of laser shot counts to update for this block       
    
    """
    for i in range(midZenithBins.shape[0]):        
        if (midZenithBins[i] > minimumZenith) and (midZenithBins[i] <= maximumZenith):            
            lowerzenith = midZenithBins[i] - zenithBinSize / 2
            upperzenith = midZenithBins[i] + zenithBinSize / 2
            for j in range(pulseZenith.shape[0]):
                if (pulseZenith[j] > lowerzenith) and (pulseZenith[j] <= upperzenith):
                    pulseCounts[i,0] += 1.0            
            for j in range(pulsesByPointZenith.shape[0]):
                if weights[j] > 0:
                    if (pulsesByPointZenith[j] > lowerzenith) and (pulsesByPointZenith[j] <= upperzenith):
                        k = int( (pointHeight[j] - heightBins[0]) / heightBinSize )
                        if (k >= 0) and (k < heightBins.shape[0]):
                            pointCounts[i,k] += weights[j]


@jit
def stratifyPointsByXYGrid(pointX, pointY, pointZ, gridX, gridY, gridZ, gridMask,
                           minX, maxX, minY, maxY, resolution, nbinsX):
    """
    Called by runXYStratification()
    
    Updates 1D arrays of point coordinates corresponding to a minimum Z grid
    
    Parameters:
        pointX          1D array of point X coordinates for this block
        pointY          1D array of point Y coordinates for this block
        pointZ          1D array of point Z coordinates for this block
        minX            Minimum X coordinate to consider
        maxX            Maximum X coordinate to consider
        minY            Minimum Y coordinate to consider
        maxY            Maximum Y coordinate to consider
        resolution      Spatial resolution of out XY grid
        nbinsX          Number of X bins in the output grid 
        
    Returns:
        gridX           A 1D array representation of a 2D grid of minumum Z point X coordinates
        gridY           A 1D array representation of a 2D grid of minumum Z point Y coordinates
        gridZ           A 1D array representation of a 2D grid of minumum Z point Z coordinates
        gridMask        A 1D array representation of a 2D bool grid of missing values
    
    """
    halfextentX = (maxX - minX) / 2
    halfextentY = (maxY - minY) / 2
    for i in range(pointZ.shape[0]):
        if (pointX[i] >= minX) and (pointX[i] <= maxX) and (pointY[i] >= minY) and (pointY[i] <= maxY):
            j = int( (pointY[i] - (-halfextentY) ) / resolution) * nbinsX + int( (pointX[i] - (-halfextentX) ) / resolution)
            if (j >= 0) and (j < gridZ.shape[0]):
                if not gridMask[j]:
                    if pointZ[i] < gridZ[j]:
                        gridX[j] = pointX[i]
                        gridY[j] = pointY[i]
                        gridZ[j] = pointZ[i]
                else:
                    gridX[j] = pointX[i]
                    gridY[j] = pointY[i]
                    gridZ[j] = pointZ[i]
                    gridMask[j] = False
    
            
def runXYStratification(data, otherargs):
    """
    Derive a minimum Z surface following plane correction procedures outlined in Calders et al. (2014)
    """
    pointcolnames = ['X','Y','Z']
    halfextent = otherargs.gridsize / 2.0
    
    for indata in data.inList:
        
        points = indata.getPoints(colNames=pointcolnames)
        
        stratifyPointsByXYGrid(points['X'], points['Y'], points['Z'], otherargs.xgrid, otherargs.ygrid, otherargs.zgrid, 
            otherargs.gridmask, -halfextent, halfextent, -halfextent, halfextent, otherargs.gridbinsize, otherargs.gridsize)


def runZenithHeightStratification(data, otherargs):
    """
    Derive Pgap(z) profiles following vertical profile procedures outlined in Calders et al. (2014)
    """
    if otherargs.planecorrection:
        pointcolnames = ['X','Y','Z','CLASSIFICATION','RETURN_NUMBER']
    else:
        pointcolnames = [otherargs.heightcol,'CLASSIFICATION','RETURN_NUMBER']
    pulsecolnames = ['NUMBER_OF_RETURNS','ZENITH']
    
    for i,indata in enumerate(data.inList):
        
        points = indata.getPoints(colNames=pointcolnames)
        pulses = indata.getPulses(colNames=pulsecolnames)
        pulsesByPoint = numpy.ma.repeat(pulses, pulses['NUMBER_OF_RETURNS'])
        
        if otherargs.weighted:
            weights = points['RETURN_NUMBER'] / pulsesByPoint['NUMBER_OF_RETURNS'].astype(numpy.float32)
        else:
            weights = numpy.array(points['RETURN_NUMBER'] == 1, dtype=numpy.float32)
        
        if otherargs.planecorrection:
            pointHeights = points['Z'] - (otherargs.planefit["Parameters"][1] * points['X'] + 
                otherargs.planefit["Parameters"][2] * points['Y'] + otherargs.planefit["Parameters"][0])
        else:
            pointHeights = points[otherargs.heightcol]
        
        stratifyPointsByZenithHeight(otherargs.zenith,otherargs.minzenith[i],otherargs.maxzenith[i],
            otherargs.zenithbinsize,pulses['ZENITH'],pulsesByPoint['ZENITH'],pointHeights,
            otherargs.height,otherargs.heightbinsize,otherargs.counts,otherargs.pulses,weights)


def calcLinearPlantProfiles(height, heightbinsize, zenith, pgapz):
    """
    Calculate the linear model PAI/PAVD (see Jupp et al., 20009)
    """ 
    kthetal = -numpy.log(pgapz)
    xtheta = 2 * numpy.tan(zenith) / numpy.pi
    paiv = numpy.zeros(pgapz.shape[1])
    paih = numpy.zeros(pgapz.shape[1])
    for i,h in enumerate(height):    
        a = numpy.vstack([xtheta, numpy.ones(xtheta.size)]).T
        y = kthetal[:,i]
        if numpy.any(y):
            lv, lh = numpy.linalg.lstsq(a, y)[0]        
            paiv[i] = lv
            paih[i] = lh
    
    pai = paiv + paih
    pavd = numpy.gradient(pai, heightbinsize)
    
    mla = numpy.degrees( numpy.arctan2(paiv,paih) )
    
    return pai,pavd,mla


def calcHingePlantProfiles(heightbinsize, zenith, pgapz):
    """
    Calculate the hinge angle PAI/PAVD (see Jupp et al., 20009)
    """       
    hingeindex = numpy.argmin(numpy.abs(zenith - numpy.arctan(numpy.pi / 2)))
    pai = -1.1 * numpy.log(pgapz[hingeindex,:])
    pavd = numpy.gradient(pai, heightbinsize)
    
    return pai,pavd
    
                      
def calcSolidAnglePlantProfiles(zenith, pgapz, heightbinsize, zenithbinsize, pai=None):
    """
    Calculate the Jupp et al. (2009) solid angle weighted PAI/PAVD
    """
    w = 2 * numpy.pi * numpy.sin(zenith) * zenithbinsize
    wn = w / numpy.sum(w[pgapz[:,-1] < 1])        
    ratio = numpy.zeros(pgapz.shape[1])
    
    for i in range(zenith.size):
        if (pgapz[i,-1] < 1):
            ratio += wn[i] * numpy.log(pgapz[i,:]) / numpy.log(pgapz[i,-1])
    
    if pai is None:
        hingeindex = numpy.argmin(numpy.abs(zenith - numpy.arctan(numpy.pi / 2)))
        pai = -1.1 * numpy.log(pgapz[hingeindex,-1])
    
    pai = pai * ratio
    pavd = numpy.gradient(pai, heightbinsize)
    
    return pai,pavd

    
def writeProfiles(outfile, zenith, height, pgapz, lpp_pai, lpp_pavd, lpp_mla, sapp_pai, sapp_pavd):
    """
    Write out the vertical profiles to file
    """  
    csvobj = open(outfile, "w")
    
    headerstr1 = ["vz%04i"%(ring*100) for ring in zenith]
    headerstr2 = ["linearPAI","linearPAVD","linearMLA","hingePAI","juppPAVD"]    
    csvobj.write("%s,%s,%s\n" % ("height",",".join(headerstr1),",".join(headerstr2)))
    
    for i in range(height.shape[0]):
        valstr1 = ["%.4f" % j for j in pgapz[:,i]]
        
        canopyz = [lpp_pai[i],lpp_pavd[i],lpp_mla[i],sapp_pai[i],sapp_pavd[i]]
        valstr2 = ["%.4f" % j for j in canopyz]
        
        csvobj.write("%.4f,%s,%s\n" % (height[i], ",".join(valstr1), ",".join(valstr2)))
    
    csvobj.close()


def planeFitHubers(x, y, z, reportfile=None):
    """
    Plane fitting (Huber's T norm with median absolute deviation scaling)
    Weighting by 1 / point range yet to be implemented in statsmodels.api
    """    
    xy = numpy.vstack((x,y)).T
    xy = sm.add_constant(xy)
    huber_t = sm.RLM(z, xy, M=sm.robust.norms.HuberT())
    huber_results = huber_t.fit()
            
    outdictn = collections.OrderedDict()
    outdictn["Parameters"] = huber_results.params        
    outdictn["Summary"] = huber_results.summary(yname='Z', xname=['Intercept','X','Y'])
    outdictn["Slope"] = numpy.degrees( numpy.arctan(numpy.sqrt(outdictn["Parameters"][1]**2 + outdictn["Parameters"][2]**2)) )
    outdictn["Aspect"] = numpy.degrees( numpy.arctan(outdictn["Parameters"][1] / outdictn["Parameters"][2]) )
    
    if reportfile is not None:
        f = open(reportfile,'w')
        for k,v in outdictn.items():
            f.write("%s:\n%s\n" % (k,v))
        f.close()
       
    return outdictn
