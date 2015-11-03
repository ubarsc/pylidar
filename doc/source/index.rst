..  _contents:

PyLidar
===================================

Introduction
------------

A set of Python modules which makes it easy to write lidar
processing code in Python. Built on top of `RIOS <http://rioshome.org/>`_
it handles the details of opening and closing files, checking alignment of projection and 
grid, stepping through the data in small blocks, etc., 
allowing the programmer to concentrate on the processing involved. 
It is licensed under GPL 3.

See :doc:`spdv4format` for description of the SPD V4 file format. 

See the `Array Visualisation <https://bitbucket.org/chchrsc/pylidar/wiki/arrays>`_ page to understand how numpy 
arrays are used in PyLidar.

Example
-------

::

    """
    Creates a Raster output file from the minimum 'Z' values of 
    the points in each bin
    """
    import numpy
    from pylidar import lidarprocessor

    dataFiles = lidarprocessor.DataFiles()
    dataFiles.input1 = lidarprocessor.LidarFile(infile, lidarprocessor.READ)
    dataFiles.imageOut1 = lidarprocessor.ImageFile(imageFile, lidarprocessor.CREATE)
    
    def writeImageFunc(data):
        zValues = data.input1.getPointsByBins(colNames='Z')
        (maxPts, nRows, nCols) = zValues.shape
        nullval = 0
        if maxPts > 0:
            minZ = zValues.min(axis=0)
            stack = numpy.ma.expand_dims(minZ, axis=0)
        else:
            stack = numpy.empty((1, nRows, nCols), dtype=zValues.dtype)
            stack.fill(nullval)
        data.imageOut1.setData(stack)

    lidarprocessor.doProcessing(writeImageFunc, dataFiles)

See :doc:`processorexamples` for more information.

Processing
-----------

.. toctree::
   :maxdepth: 2

   userclasses
   lidarprocessor
   toolbox/toolbox

Drivers
---------

.. toctree::
   :maxdepth: 2

   basedriver
   gdaldriver
   lidarformats/generic
   lidarformats/spdv3
   lidarformats/spdv4
   lidarformats/riegl
   lidarformats/h5space
   lidarformats/gridindexutils


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

