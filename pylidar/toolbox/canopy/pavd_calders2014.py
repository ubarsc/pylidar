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


def calcLinearPlantProfiles(height, zenith, pgap):
   """
   Calculate the linear model PAI/PAVD
   """ 
   kthetal = -numpy.log(pgap)
   xtheta = 2 * numpy.tan(zenith) / numpy.pi
   paiv = numpy.zeros(height.size)
   paih = numpy.zeros(height.size)
   for i,h in enumerate(height):    
       a = numpy.vstack([xtheta, numpy.ones(xtheta.size)]).t
       y = kthetal[:,i]
       if y.any():
           lv, lh = numpy.linalg.lstsq(a, y)[0]        
           paiv[i] = lv
           paih[i] = lh
   
   pai = paiv + paih
   pavd = deriv(self.heights,self.pai)
   mla = 90 - numpy.degrees(numpy.arctan2(paiv,paih))
   
   return pai,pavd,mla


def calcHingePlantProfiles(height, zenith, pgap):
    """
    Calculate the hinge angle PAI/PAVD
    """       
    hingeindex = numpy.argmin(numpy.abs(zenith - numpy.arctan(numpy.pi / 2)))
    pai = -1.1 * numpy.log(pgap[hingeindex,:])       
    pavd = deriv(height,pai)
    
    return pai,pavd
    
                      
def calcSolidAnglePlantProfiles(height, zenith, pgap, zenithbinsize, pai=None):
    """
    Calculate the Jupp et al. (2009) solid angle weighted PAI/PAVD
    """
    w = 2 * numpy.pi * numpy.sin(zenith) * zenithbinsize
    wn = w / numpy.sum(w[pgap[:,-1] < 1])        
    ratio = numpy.zeros(height.size)
    
    for i in range(zenith.size):
        if (pgap[i,-1] < 1):
            ratio += wn[i] * numpy.log(pgap[i,:]) / numpy.log(pgap[i,-1])
    
    if pai is None:
        hingeindex = numpy.argmin(numpy.abs(zenith - numpy.arctan(numpy.pi / 2)))
        pai = -1.1 * numpy.log(pgap[hingeindex,-1])
    
    pai = pai * ratio
    pavd = deriv(height,pai)
    
    return pai, pavd       


def deriv(x,y):
    """
    IDL's numerical differentiation using 3-point, Lagrangian interpolation
    df/dx = y0*(2x-x1-x2)/(x01*x02)+y1*(2x-x0-x2)/(x10*x12)+y2*(2x-x0-x1)/(x20*x21)
    where: x01 = x0-x1, x02 = x0-x2, x12 = x1-x2, etc.
    """
    
    x12 = x - numpy.roll(x,-1) #x1 - x2
    x01 = numpy.roll(x,1) - x #x0 - x1
    x02 = numpy.roll(x,1) - numpy.roll(x,-1) #x0 - x2

    d = numpy.roll(y,1) * (x12 / (x01*x02)) + y * (1.0/x12 - 1.0/x01) - numpy.roll(y,-1) * (x01 / (x02 * x12)) # Middle points
    d[0] = y[0] * (x01[1]+x02[1])/(x01[1]*x02[1]) - y[1] * x02[1]/(x01[1]*x12[1]) + y[2] * x01[1]/(x02[1]*x12[1]) # First point
    d[y.size-1] = -y[y.size-3] * x12[y.size-2]/(x01[y.size-2]*x02[y.size-2]) + \
    y[y.size-2] * x02[y.size-2]/(x01[y.size-2]*x12[y.size-2]) - y[y.size-1] * \
    (x02[y.size-2]+x12[y.size-2]) / (x02[y.size-2]*x12[y.size-2]) # Last point

    return d
