#!/usr/bin/env python

"""
Install script for PyLidar
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
from __future__ import print_function

import os
import sys
# If we fail to import the numpy version of setup, still try to proceed, as it is possibly
# because we are being run by ReadTheDocs, and so we just need to be able to generate documentation. 
try:
    from numpy.distutils.core import setup, Extension
    withExtensions = True
except ImportError:
    from distutils.core import setup
    withExtensions = False

import pylidar

# use the latest numpy API
NUMPY_MACROS = ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')

# Are we installing the command line scripts?
# this is an experimental option for users who are
# using the Python entry point feature of setuptools and Conda instead
NO_INSTALL_CMDLINE = int(os.getenv('PYLIDAR_NOCMDLINE', '0')) > 0

def getExtraCXXFlags():
    """
    Looks at the $PYLIDAR_CXX_FLAGS environment variable.
    If it exists, this function returns a list of flags
    to be passed as the extra_compile_args argument to
    the Extension constructor.
    Otherwise None.
    """
    if 'PYLIDAR_CXX_FLAGS' in os.environ:
        return os.environ['PYLIDAR_CXX_FLAGS'].split()
    else:
        return None

def addRieglDriver(extModules, cxxFlags):
    """
    Decides if the Riegl driver is to be built. If so 
    adds the Extension class to extModules.
    """
    if 'RIVLIB_ROOT' in os.environ and 'RIWAVELIB_ROOT' in os.environ:
        print('Building Riegl Extension...')
        rivlibRoot = os.environ['RIVLIB_ROOT']
        riwavelibRoot = os.environ['RIWAVELIB_ROOT']
        rivlibs = ['scanlib-mt', 'riboost_chrono-mt', 
                     'riboost_date_time-mt', 'riboost_filesystem-mt', 
                     'riboost_regex-mt', 'riboost_system-mt', 
                     'riboost_thread-mt']
        riwavelibs = ['wfmifc-mt']
        # on Windows the libs do not follow the normal naming convention
        # and start with 'lib'. On Linux the compiler prepends this automatically
        # but on Windows we need to do it manually
        if sys.platform == 'win32':
            rivlibs = ['lib' + name for name in rivlibs]
            
        # all the libs
        libs = rivlibs + riwavelibs
        
        defines = getRieglWaveLibVersion(riwavelibRoot, riwavelibs[0])
        defines.extend([NUMPY_MACROS])
        
        rieglModule = Extension(name='pylidar.lidarformats._riegl', 
                define_macros=defines,
                sources=['src/riegl.cpp', 'src/pylidar.c'],
                include_dirs=[os.path.join(rivlibRoot, 'include'),
                                os.path.join(riwavelibRoot, 'include')],
                extra_compile_args=cxxFlags,
                libraries=libs,
                library_dirs=[os.path.join(rivlibRoot, 'lib'),
                                os.path.join(riwavelibRoot, 'lib')])
                 
        extModules.append(rieglModule)
    else:
        print('Riegl Libraries not found.')
        print('If installed set $RIVLIB_ROOT to the install location of RiVLib')
        print('and $RIWAVELIB_ROOT to the install location of the waveform extraction library (riwavelib)')
        
def getRieglWaveLibVersion(riwavelibRoot, libname):
    """
    Because we cannot distribute the wfmifc-mt library, we need
    to check that the major version at compile time matches the 
    version the user has at runtime. We do this by getting the
    version now and setting it as a #define. The library can then
    use the #define to check at runtime.
    
    Unfortunately the headers don't give us this information.
    
    """
    import ctypes
    if sys.platform == 'win32':
        libname = os.path.join(riwavelibRoot, 'lib', libname + '.dll')
    elif sys.platform == 'darwin':
        libname = os.path.join(riwavelibRoot, 'lib', 'lib' + libname + '.dylib')
    else:
        libname = os.path.join(riwavelibRoot, 'lib', 'lib' + libname + '.so')
    wfm = ctypes.cdll.LoadLibrary(libname)
    
    major = ctypes.c_ushort()
    minor = ctypes.c_ushort()
    version = ctypes.c_char_p()
    tag = ctypes.c_char_p()
    wfm.fwifc_get_library_version(ctypes.byref(major), ctypes.byref(minor),
                ctypes.byref(version), ctypes.byref(tag))
                
    return [("RIEGL_WFM_MAJOR", str(major.value)), 
            ("RIEGL_WFM_MINOR", str(minor.value))]
                
        
def addLasDriver(extModules, cxxFlags):
    """
    Decides if the Las driver is to be built. If so
    adds the Extension class to extModules.
    """
    import glob
    if 'LASTOOLS_ROOT' in os.environ:
        print('Building Las Extension...')
        lastoolsRoot = os.environ['LASTOOLS_ROOT']
        # do extra check of name of lib. By default it is 'liblas'
        # but potentially renamed to 'laslib' so it doesn't conflict with 
        # Howard Butler's lib of the same name. lastools on the Conda
        # 'rios' channel does this. 
        if sys.platform == 'win32':
            prefix = ''
        else:
            prefix = 'lib'
        laslibwildcard = os.path.join(lastoolsRoot, 'lib', prefix + 'laslib.*')
        liblaswildcard = os.path.join(lastoolsRoot, 'lib', prefix + 'las.*')
        if len(glob.glob(laslibwildcard)) > 0:
            print('Found laslib')
            lasLib = 'laslib'
        elif len(glob.glob(liblaswildcard)) > 0:
            print('Found liblas')
            lasLib = 'las'
        else:
            print('Found neither laslib not liblas')
            print('Assuming liblas')
            lasLib = 'las'

        lasModule = Extension(name='pylidar.lidarformats._las',
                sources=['src/las.cpp', 'src/pylidar.c'],
                include_dirs=[os.path.join(lastoolsRoot, 'include')],
                extra_compile_args=cxxFlags,
                define_macros = [NUMPY_MACROS],
                libraries=[lasLib],
                library_dirs=[os.path.join(lastoolsRoot, 'lib')])
                
        extModules.append(lasModule)
    else:
        print('Las library not found.')
        print('If installed set $LASTOOLS_ROOT to the install location of lastools https://github.com/LAStools/LAStools')


def addASCIIDriver(extModules, cxxFlags):
    """
    Decides if the ASCII driver is to be built. If so
    adds the Extension class to extModules.
    """
    print('Building ASCII Extension...')
    includeDirs = None
    libs = None
    libraryDirs = None
    defineMacros = [NUMPY_MACROS]
    if 'ZLIB_ROOT' in os.environ:
        # build for zlib
        zlibRoot = os.environ['ZLIB_ROOT']
        includeDirs = [os.path.join(zlibRoot, 'include')]
        libs = ['z']
        libraryDirs = [os.path.join(zlibRoot, 'lib')]
        defineMacros.append(("HAVE_ZLIB", "1"))
    else:
        print('zlib library not found.')
        print('If installed set $ZLIB_ROOT to the install location of zlib')
        print("gzip compressed files won't be read")

    asciiModule = Extension(name='pylidar.lidarformats._ascii',
        sources=['src/ascii.cpp', 'src/pylidar.c'],
        include_dirs=includeDirs,
        extra_compile_args=cxxFlags,
        define_macros=defineMacros,
        libraries=libs,
        library_dirs=libraryDirs)

    extModules.append(asciiModule)

def addAdvIndexing(extModules, cxxFlags):
    """
    Decides if the Advanced Indexing is to be built. If so
    adds the Extension class to extModules.
    """
    if 'LIBSPATIALINDEX_ROOT' in os.environ:
        print('Building Advanced Indexing Extension...')
        libspatialindexRoot = os.environ['LIBSPATIALINDEX_ROOT']

        advIdxModule = Extension(name='pylidar.lidarformats._advindex',
            sources=['src/advindex.cpp', 'src/pylidar.c'],
            include_dirs=[os.path.join(libspatialindexRoot, 'include')],
            extra_compile_args=cxxFlags,
            define_macros=[NUMPY_MACROS],
            libraries=['spatialindex_c'],
            library_dirs=[os.path.join(libspatialindexRoot, 'lib')])

        extModules.append(advIdxModule)
    else:
        print('libspatialindex library not found.')
        print('If installed set $LIBSPATIALINDEX_ROOT to the install location of libspatialindex https://libspatialindex.github.io/')

def addInsidePoly(extModules):
    """
    Adds the insidepoly C toolbox module. Currently requires GDAL to be 
    present. Could be ignored if GDAL not available, but that sounds confusing
    so left it as compulsory for now.
    """
    extraargs = {}
    # don't use the deprecated numpy api
    extraargs['define_macros'] = [NUMPY_MACROS]

    if sys.platform == 'win32':
        # Windows - rely on %GDAL_HOME% being set and set 
        # paths appropriately
        gdalhome = os.getenv('GDAL_HOME')
        if gdalhome is None:
            raise SystemExit("need to define %GDAL_HOME%")
        extraargs['include_dirs'] = [os.path.join(gdalhome, 'include')]
        extraargs['library_dirs'] = [os.path.join(gdalhome, 'lib')]
        extraargs['libraries'] = ['gdal_i']
    else:
        # Unix - can do better with actual flags using gdal-config
        import subprocess
        try:
            cflags = subprocess.check_output(['gdal-config', '--cflags'])
            if sys.version_info[0] >= 3:
                cflags = cflags.decode()
            extraargs['extra_compile_args'] = cflags.strip().split()

            ldflags = subprocess.check_output(['gdal-config', '--libs'])
            if sys.version_info[0] >= 3:
                ldflags = ldflags.decode()
            extraargs['extra_link_args'] = ldflags.strip().split()
        except OSError:
            raise SystemExit("can't find gdal-config - GDAL development files need to be installed")

    extraargs['name'] = 'pylidar.toolbox.insidepoly'
    extraargs['sources'] = ['src/insidepoly.c']
    print('Building InsidePoly Toolbox Extension...')

    insidePolyModule = Extension(**extraargs)
    extModules.append(insidePolyModule)

# get any C++ flags
cxxFlags = getExtraCXXFlags()
# work out if we need to build any of the C/C++ extension
# modules
externalModules = []
if withExtensions:
    addRieglDriver(externalModules, cxxFlags)
    addLasDriver(externalModules, cxxFlags)
    addASCIIDriver(externalModules, cxxFlags)
    # Advanced indexing commented out for now
    # wasn't useful, and causing problems for some installs
    #addAdvIndexing(externalModules, cxxFlags)
    addInsidePoly(externalModules)

if NO_INSTALL_CMDLINE:
    scriptList = None
else:
    scriptList = ['bin/pylidar_translate', 'bin/pylidar_info', 
            'bin/pylidar_index', 'bin/pylidar_tile', 'bin/pylidar_rasterize',
            'bin/pylidar_test', 'bin/pylidar_canopy']

setup(name='pylidar',
      version=pylidar.PYLIDAR_VERSION,
      ext_modules=externalModules,
      description='Tools for simplifying LiDAR data I/O and tools for processing.',
      packages=['pylidar', 'pylidar/lidarformats', 'pylidar/toolbox', 
                'pylidar/toolbox/grdfilters', 'pylidar/toolbox/indexing',
                'pylidar/toolbox/translate', 'pylidar.toolbox.cmdline', 
                'pylidar/testing', 'pylidar/toolbox/canopy'],
      scripts=scriptList,
      license='LICENSE.txt', 
      url='http://pylidar.org/',
      classifiers=['Intended Audience :: Developers',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.2',
          'Programming Language :: Python :: 3.3',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5'])
      
