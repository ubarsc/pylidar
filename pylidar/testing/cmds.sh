#!/bin/sh

# testsuite1
pylidar_translate -i apl1dr_x509000ys6945000z56_2009_ba1m6_pbrisba.las \
    -o testsuite1.spd --epsg 28356 -f SPDV4 --pulseindex FIRST_RETURN
pylidar_index -i testsuite1.spd -o testsuite1_idx.spd -r 2.0 
pylidar_rasterize -i testsuite1_idx.spd -o testsuite1.img -f numpy.ma.min -a Z

# testsuite2 
pylidar_translate -i apl1dr_x510000ys6945000z56_2009_ba1m6_pbrisba.las \
    -o testsuite2.spd --epsg 28356 -f SPDV4 --pulseindex FIRST_RETURN
pylidar_index -i testsuite2.spd -o testsuite2_idx.spd -r 2.0 
pylidar_rasterize -i testsuite1_idx.spd testsuite2_idx.spd \
    -o testsuite2.img -f numpy.ma.min -a Z --binsize 3.0 --footprint UNION

# testsuite3
laszip -i apl1dr_x509000ys6945000z56_2009_ba1m6_pbrisba.las \
    -o apl1dr_x509000ys6945000z56_2009_ba1m6_pbrisba_zip.laz
lasindex -i apl1dr_x509000ys6945000z56_2009_ba1m6_pbrisba_zip.laz
pylidar_translate -i apl1dr_x509000ys6945000z56_2009_ba1m6_pbrisba_zip.laz \
    -o testsuite3.spd --epsg 28356 -f SPDV4 --pulseindex FIRST_RETURN \
    --spatial --binsize=2.0
pylidar_rasterize -i testsuite3.spd -o testsuite3.img -f numpy.ma.min -a Z 

# testsuite4
python -c "from pylidar.testing.testsuite4 import run;run('.', '.')"

cd ..
tar cvfz testdata_1.tar.gz testdata

