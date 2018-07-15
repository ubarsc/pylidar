====================================
Command Line Examples: pylidar_index
====================================
    
--------------------------------------------------------
Creating a Spatial Index with pylidar_index (deprecated)
--------------------------------------------------------

Once you have converted your data to SPV4 format, you can use this
utility to spatially index the data. Once it is spatially indexed, 
you can process it spatially.

Run "pylidar_index -h" to obtain full help. The basic usage is to specify input and output files
like this::
    
    pylidar_index --input data.spdv4 --output indexed.spdv4

Important options that you should consider overriding are:

    * The resolution that the spatial index is calculated on (the -r or --resolution flag). This will control how much data goes in each bin.
    * The type of index (the --indextype flag). This controls what the spatial index is calculated on. CARTESIAN or SCAN is usual for airborne, but SPHERICAL for TLS.
    * The size of the blocks that the spatial indexing will use (the -b or --blocksize flag).This will determine the amount of memory used. The smaller the blocks the less memory will be used. The value will be in the units that the spatial index is being calculated in. By default pylidar_index uses 200.
    * The temporary directory to create the temprary files in. By default this is the current directory, but you may want to change this if you do not have enough space there.

