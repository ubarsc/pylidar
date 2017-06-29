
"""
Simple testsuite that checks we can import an LVIS ASCII file
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
from . import utils
from pylidar.lidarformats import generic
from pylidar.toolbox.translate.ascii2spdv4 import translate

INPUT_ASCII = 'testsuite20.csv' # from the PulseWaves github repo
IMPORTED_SPD = 'testsuite20.spd'

COLTYPES = [('LFID', 'UINT32'), ('SHOTNUMBER', 'UINT32'), ('X', 'UINT32'), 
('Y', 'UINT32'), ('Z', 'UINT32'), ('ZB', 'FLOAT32'), ('RG', 'FLOAT32'), 
('RV', 'FLOAT32'), ('SCANLINE', 'UINT32'), ('SCANLINE_IDX', 'UINT16'),
('X_IDX', 'UINT32'), ('Y_IDX', 'UINT32'), ('ZT', 'FLOAT32'), 
('CLASSIFICATION', 'UINT8'), ('FLAG3', 'UINT8'), ('GRANGE', 'FLOAT32'),
('INCIDENTANGLE', 'FLOAT32'), ('RH10', 'FLOAT32'), ('RH15', 'FLOAT32'),
('RH20', 'FLOAT32'), ('RH25', 'FLOAT32'), ('RH30', 'FLOAT32'), 
('RH35', 'FLOAT32'), ('RH40', 'FLOAT32'), ('RH45', 'FLOAT32'),
('RH50', 'FLOAT32'), ('RH55', 'FLOAT32'), ('RH60', 'FLOAT32'), 
('RH65', 'FLOAT32'),
('RH70', 'FLOAT32'), ('RH75', 'FLOAT32'), ('RH80', 'FLOAT32'), 
('RH85', 'FLOAT32'), ('RH90', 'FLOAT32'), ('RH95', 'FLOAT32'), 
('RH96', 'FLOAT32'), ('RH97', 'FLOAT32'), ('RH98', 'FLOAT32'), 
('RH99', 'FLOAT32'), ('RH100', 'FLOAT32'), ('CC_Z00', 'FLOAT32'), 
('CC_Z01', 'FLOAT32'), ('CC_Z02', 'FLOAT32'), ('CC_Z03', 'FLOAT32'), 
('CC_Z04', 'FLOAT32'), ('CC_Z05', 'FLOAT32'), ('CC_Z06', 'FLOAT32'), 
('CC_Z07', 'FLOAT32'), ('CC_Z08', 'FLOAT32'), ('CC_Z09', 'FLOAT32'), 
('CC_Z10', 'FLOAT32'), ('CC_Z11', 'FLOAT32'), ('CC_Z12', 'FLOAT32'), 
('CC_Z13', 'FLOAT32'), ('CC_Z14', 'FLOAT32'), ('CC_Z15', 'FLOAT32'), 
('CC_Z16', 'FLOAT32'), ('CC_Z17', 'FLOAT32'), ('CC_Z18', 'FLOAT32'), 
('CC_Z19', 'FLOAT32'), ('CC_Z20', 'FLOAT32'), ('CC_Z21', 'FLOAT32'), 
('CC_Z22', 'FLOAT32'), ('CC_Z23', 'FLOAT32'), ('CC_Z24', 'FLOAT32'), 
('CC_Z25', 'FLOAT32'), ('CC_Z26', 'FLOAT32'), ('CC_Z27', 'FLOAT32'), 
('CC_Z28', 'FLOAT32'), ('CC_Z29', 'FLOAT32'), ('CC_Z30', 'FLOAT32'), 
('CC_Z31', 'FLOAT32'), ('CC_Z32', 'FLOAT32'), ('CC_Z33', 'FLOAT32'), 
('CC_Z34', 'FLOAT32'), ('CC_Z35', 'FLOAT32'), ('CC_Z36', 'FLOAT32'), 
('CC_Z37', 'FLOAT32'), ('CC_Z38', 'FLOAT32'), ('CC_Z39', 'FLOAT32'), 
('CC_Z40', 'FLOAT32'), ('CC_Z41', 'FLOAT32'), ('CC_Z42', 'FLOAT32'), 
('CC_Z43', 'FLOAT32'), ('CC_Z44', 'FLOAT32'), ('CC_Z45', 'FLOAT32'), 
('CC_Z46', 'FLOAT32'), ('CC_Z47', 'FLOAT32'), ('CC_Z48', 'FLOAT32'), 
('CC_Z49', 'FLOAT32'), ('CC_Z50', 'FLOAT32'), ('CC_Z51', 'FLOAT32'), 
('CC_Z52', 'FLOAT32'), ('CC_Z53', 'FLOAT32'), ('CC_Z54', 'FLOAT32'), 
('CC_Z55', 'FLOAT32'), ('CC_Z56', 'FLOAT32'), ('CC_Z57', 'FLOAT32'), 
('CC_Z58', 'FLOAT32'), ('CC_Z59', 'FLOAT32'), ('CC_Z60', 'FLOAT32'), 
('CC_Z61', 'FLOAT32'), ('CC_Z62', 'FLOAT32'), ('CC_Z63', 'FLOAT32'), 
('CC_Z64', 'FLOAT32'), ('CC_Z65', 'FLOAT32'), ('CC_Z66', 'FLOAT32'), 
('CC_Z67', 'FLOAT32'), ('CC_Z68', 'FLOAT32'), ('CC_Z69', 'FLOAT32')]

PULSECOLS = ['LFID','SHOTNUMBER','ZB','RG','RV','SCANLINE','SCANLINE_IDX',
'X_IDX','Y_IDX','ZT','FLAG3','GRANGE','INCIDENTANGLE','RH10','RH15','RH20',
'RH25','RH30','RH35','RH40','RH45','RH50','RH55','RH60','RH65','RH70','RH75',
'RH80','RH85','RH90','RH95','RH96','RH97','RH98','RH99','RH100','CC_Z00',
'CC_Z01','CC_Z02','CC_Z03','CC_Z04','CC_Z05','CC_Z06','CC_Z07','CC_Z08',
'CC_Z09','CC_Z10','CC_Z11','CC_Z12','CC_Z13','CC_Z14','CC_Z15','CC_Z16',
'CC_Z17','CC_Z18','CC_Z19','CC_Z20','CC_Z21','CC_Z22','CC_Z23','CC_Z24',
'CC_Z25','CC_Z26','CC_Z27','CC_Z28','CC_Z29','CC_Z30','CC_Z31','CC_Z32',
'CC_Z33','CC_Z34','CC_Z35','CC_Z36','CC_Z37','CC_Z38','CC_Z39','CC_Z40',
'CC_Z41','CC_Z42','CC_Z43','CC_Z44','CC_Z45','CC_Z46','CC_Z47','CC_Z48',
'CC_Z49','CC_Z50','CC_Z51','CC_Z52','CC_Z53','CC_Z54','CC_Z55','CC_Z56',
'CC_Z57','CC_Z58','CC_Z59','CC_Z60','CC_Z61','CC_Z62','CC_Z63','CC_Z64',
'CC_Z65','CC_Z66','CC_Z67','CC_Z68','CC_Z69']

SCALING = [('PULSE', 'LFID', 'UINT32', 1.0, 0.0), ('PULSE', 'SHOTNUMBER', 'UINT32', 1.0, 0.0), 
('POINT', 'X', 'DFLT', 100.000000, -500000.000000), ('POINT', 'Y', 'DFLT', 100.000000, -10000000.000000), 
('POINT', 'Z', 'DFLT', 100.000000, -500.000000), ('PULSE', 'ZB', 'FLOAT32', 1.0, 0.0), 
('PULSE', 'RG', 'FLOAT32', 1.0, 0.0), ('PULSE', 'RV', 'FLOAT32', 1.0, 0.0), 
('PULSE', 'SCANLINE', 'DFLT', 1.0, 0.0), ('PULSE', 'SCANLINE_IDX', 'DFLT', 1.0, 0.0), 
('PULSE', 'X_IDX', 'DFLT', 1.0, 0.0), ('PULSE', 'Y_IDX', 'DFLT', 1.0, 0.0), 
('PULSE', 'ZT', 'FLOAT32', 1.0, 0.0), ('POINT', 'CLASSIFICATION', 'DFLT', 1.000000, 0.000000), 
('PULSE', 'FLAG3', 'UINT8', 1.0, 0.0), ('PULSE', 'GRANGE', 'FLOAT32', 1.0, 0.0), 
('PULSE', 'INCIDENTANGLE', 'FLOAT32', 1.0, 0.0), ('PULSE', 'RH10', 'FLOAT32', 1.0, 0.0), 
('PULSE', 'RH15', 'FLOAT32', 1.0, 0.0), ('PULSE', 'RH20', 'FLOAT32', 1.0, 0.0), 
('PULSE', 'RH25', 'FLOAT32', 1.0, 0.0), ('PULSE', 'RH30', 'FLOAT32', 1.0, 0.0), 
('PULSE', 'RH35', 'FLOAT32', 1.0, 0.0), ('PULSE', 'RH40', 'FLOAT32', 1.0, 0.0), 
('PULSE', 'RH45', 'FLOAT32', 1.0, 0.0), ('PULSE', 'RH50', 'FLOAT32', 1.0, 0.0), 
('PULSE', 'RH55', 'FLOAT32', 1.0, 0.0), ('PULSE', 'RH60', 'FLOAT32', 1.0, 0.0), 
('PULSE', 'RH65', 'FLOAT32', 1.0, 0.0), ('PULSE', 'RH70', 'FLOAT32', 1.0, 0.0), 
('PULSE', 'RH75', 'FLOAT32', 1.0, 0.0), ('PULSE', 'RH80', 'FLOAT32', 1.0, 0.0), 
('PULSE', 'RH85', 'FLOAT32', 1.0, 0.0), ('PULSE', 'RH90', 'FLOAT32', 1.0, 0.0), 
('PULSE', 'RH95', 'FLOAT32', 1.0, 0.0), ('PULSE', 'RH96', 'FLOAT32', 1.0, 0.0), 
('PULSE', 'RH97', 'FLOAT32', 1.0, 0.0), ('PULSE', 'RH98', 'FLOAT32', 1.0, 0.0), 
('PULSE', 'RH99', 'FLOAT32', 1.0, 0.0), ('PULSE', 'RH100', 'FLOAT32', 1.0, 0.0), 
('PULSE', 'CC_Z00', 'FLOAT32', 1.0, 0.0), ('PULSE', 'CC_Z01', 'FLOAT32', 1.0, 0.0), 
('PULSE', 'CC_Z02', 'FLOAT32', 1.0, 0.0), ('PULSE', 'CC_Z03', 'FLOAT32', 1.0, 0.0), 
('PULSE', 'CC_Z04', 'FLOAT32', 1.0, 0.0), ('PULSE', 'CC_Z05', 'FLOAT32', 1.0, 0.0), 
('PULSE', 'CC_Z06', 'FLOAT32', 1.0, 0.0), ('PULSE', 'CC_Z07', 'FLOAT32', 1.0, 0.0), 
('PULSE', 'CC_Z08', 'FLOAT32', 1.0, 0.0), ('PULSE', 'CC_Z09', 'FLOAT32', 1.0, 0.0), 
('PULSE', 'CC_Z10', 'FLOAT32', 1.0, 0.0), ('PULSE', 'CC_Z11', 'FLOAT32', 1.0, 0.0), 
('PULSE', 'CC_Z12', 'FLOAT32', 1.0, 0.0), ('PULSE', 'CC_Z13', 'FLOAT32', 1.0, 0.0), 
('PULSE', 'CC_Z14', 'FLOAT32', 1.0, 0.0), ('PULSE', 'CC_Z15', 'FLOAT32', 1.0, 0.0), 
('PULSE', 'CC_Z16', 'FLOAT32', 1.0, 0.0), ('PULSE', 'CC_Z17', 'FLOAT32', 1.0, 0.0), 
('PULSE', 'CC_Z18', 'FLOAT32', 1.0, 0.0), ('PULSE', 'CC_Z19', 'FLOAT32', 1.0, 0.0), 
('PULSE', 'CC_Z20', 'FLOAT32', 1.0, 0.0), ('PULSE', 'CC_Z21', 'FLOAT32', 1.0, 0.0), 
('PULSE', 'CC_Z22', 'FLOAT32', 1.0, 0.0), ('PULSE', 'CC_Z23', 'FLOAT32', 1.0, 0.0), 
('PULSE', 'CC_Z24', 'FLOAT32', 1.0, 0.0), ('PULSE', 'CC_Z25', 'FLOAT32', 1.0, 0.0), 
('PULSE', 'CC_Z26', 'FLOAT32', 1.0, 0.0), ('PULSE', 'CC_Z27', 'FLOAT32', 1.0, 0.0), 
('PULSE', 'CC_Z28', 'FLOAT32', 1.0, 0.0), ('PULSE', 'CC_Z29', 'FLOAT32', 1.0, 0.0), 
('PULSE', 'CC_Z30', 'FLOAT32', 1.0, 0.0), ('PULSE', 'CC_Z31', 'FLOAT32', 1.0, 0.0), 
('PULSE', 'CC_Z32', 'FLOAT32', 1.0, 0.0), ('PULSE', 'CC_Z33', 'FLOAT32', 1.0, 0.0), 
('PULSE', 'CC_Z34', 'FLOAT32', 1.0, 0.0), ('PULSE', 'CC_Z35', 'FLOAT32', 1.0, 0.0), 
('PULSE', 'CC_Z36', 'FLOAT32', 1.0, 0.0), ('PULSE', 'CC_Z37', 'FLOAT32', 1.0, 0.0), 
('PULSE', 'CC_Z38', 'FLOAT32', 1.0, 0.0), ('PULSE', 'CC_Z39', 'FLOAT32', 1.0, 0.0), 
('PULSE', 'CC_Z40', 'FLOAT32', 1.0, 0.0), ('PULSE', 'CC_Z41', 'FLOAT32', 1.0, 0.0), 
('PULSE', 'CC_Z42', 'FLOAT32', 1.0, 0.0), ('PULSE', 'CC_Z43', 'FLOAT32', 1.0, 0.0), 
('PULSE', 'CC_Z44', 'FLOAT32', 1.0, 0.0), ('PULSE', 'CC_Z45', 'FLOAT32', 1.0, 0.0), 
('PULSE', 'CC_Z46', 'FLOAT32', 1.0, 0.0), ('PULSE', 'CC_Z47', 'FLOAT32', 1.0, 0.0), 
('PULSE', 'CC_Z48', 'FLOAT32', 1.0, 0.0), ('PULSE', 'CC_Z49', 'FLOAT32', 1.0, 0.0), 
('PULSE', 'CC_Z50', 'FLOAT32', 1.0, 0.0), ('PULSE', 'CC_Z51', 'FLOAT32', 1.0, 0.0), 
('PULSE', 'CC_Z52', 'FLOAT32', 1.0, 0.0), ('PULSE', 'CC_Z53', 'FLOAT32', 1.0, 0.0), 
('PULSE', 'CC_Z54', 'FLOAT32', 1.0, 0.0), ('PULSE', 'CC_Z55', 'FLOAT32', 1.0, 0.0), 
('PULSE', 'CC_Z56', 'FLOAT32', 1.0, 0.0), ('PULSE', 'CC_Z57', 'FLOAT32', 1.0, 0.0), 
('PULSE', 'CC_Z58', 'FLOAT32', 1.0, 0.0), ('PULSE', 'CC_Z59', 'FLOAT32', 1.0, 0.0), 
('PULSE', 'CC_Z60', 'FLOAT32', 1.0, 0.0), ('PULSE', 'CC_Z61', 'FLOAT32', 1.0, 0.0), 
('PULSE', 'CC_Z62', 'FLOAT32', 1.0, 0.0), ('PULSE', 'CC_Z63', 'FLOAT32', 1.0, 0.0), 
('PULSE', 'CC_Z64', 'FLOAT32', 1.0, 0.0), ('PULSE', 'CC_Z65', 'FLOAT32', 1.0, 0.0), 
('PULSE', 'CC_Z66', 'FLOAT32', 1.0, 0.0), ('PULSE', 'CC_Z67', 'FLOAT32', 1.0, 0.0), 
('PULSE', 'CC_Z68', 'FLOAT32', 1.0, 0.0), ('PULSE', 'CC_Z69', 'FLOAT32', 1.0, 0.0)]

def run(oldpath, newpath):
    """
    Runs the 20th basic test suite. Tests:

    Importing an LVIS ASCII file
    """
    inputASC = os.path.join(oldpath, INPUT_ASCII)
    info = generic.getLidarFileInfo(inputASC)

    importedSPD = os.path.join(newpath, IMPORTED_SPD)
    translate(info, inputASC, importedSPD, colTypes=COLTYPES, pulseCols=PULSECOLS,
            scaling=SCALING)
    utils.compareLiDARFiles(os.path.join(oldpath, IMPORTED_SPD), importedSPD)
