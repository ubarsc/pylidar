..  _contents:

PyLidar
===================================

Introduction
------------

A set of Python modules which makes it easy to write lidar
processing code in Python. Based on `SPDLib <http://www.spdlib.org/>`_ and 
built on top of `RIOS <http://rioshome.org/>`_
it handles the details of opening and closing files, checking alignment of projection and 
grid, stepping through the data in small blocks, etc., 
allowing the programmer to concentrate on the processing involved. 
It is licensed under GPL 3.

See :doc:`spdv4format` for description of the SPD V4 file format. 

See the :doc:`arrayvisualisation` page to understand how numpy 
arrays are used in PyLidar.

Work funded by `DSITI <https://www.qld.gov.au/dsiti/>`_ and 
`OEH <http://www.environment.nsw.gov.au/>`_ through the 
`Joint Remote Sensing Research Program <https://www.gpem.uq.edu.au/jrsrp>`_.

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

.. image:: pylidar.jpg

Downloads
---------

Source code is available from `BitBucket <https://bitbucket.org/chchrsc/pylidar>`_. 
`RIOS <http://rioshome.org/>`_, `Numba <http://numba.pydata.org/>`_, `Numpy <http://www.numpy.org/>`_
and `h5py <http://h5py.org/>`_ are required dependencies. For LAS support, install
`lastools <https://github.com/LAStools/LAStools>`_ and set the LASTOOLS_ROOT environment variable
to point to it before installation. For Riegl format support, install `RiVLIB <http://www.riegl.com/index.php?id=224>`_ and
`RiWaveLIB <http://www.riegl.com/index.php?id=322>`_ and set the RIVLIB_ROOT and RIWAVELIB_ROOT environment variables
to point to them before installation.

`Conda <http://conda.pydata.org/miniconda.html#miniconda>`_ packages are available under the 'rios' channel.
Once you have installed `Conda <http://conda.pydata.org/miniconda.html#miniconda>`_, run the following commands on the 
command line to install pylidar (dependencies are obtained automatically): ::

    conda config --add channels conda-forge 
    conda config --add channels rios 
    conda create -n myenv pylidar
    source activate myenv # omit 'source' on Windows

The related `pynninterp <https://bitbucket.org/petebunting/pynninterp>`_ module is used
for some interpolation operations and can be installed via Conda also from the 'rios' channel::

    conda install pynninterp


Processing
-----------

.. toctree::
   :maxdepth: 2

   userclasses
   lidarprocessor
   toolbox/toolbox
   toolbox/indexing
   toolbox/translate

Drivers
---------

.. toctree::
   :maxdepth: 2

   basedriver
   gdaldriver
   lidarformats/generic
   lidarformats/spdv3
   lidarformats/spdv4
   lidarformats/las
   lidarformats/riegl
   lidarformats/ascii_ts
   lidarformats/h5space
   lidarformats/gridindexutils

Testing
-------
.. toctree::
   :maxdepth: 2

   testing

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

