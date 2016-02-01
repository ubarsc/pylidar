#!/usr/bin/env python

from __future__ import print_function, division

import sys
from pylidar.testing import utils

cksum = utils.calculateCheckSum(sys.argv[1])
print(cksum)
