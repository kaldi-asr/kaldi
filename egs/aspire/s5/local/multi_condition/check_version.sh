#!/bin/bash

# Script to check the tool versions necessary for the aspire recipe
python -c "
from distutils.version import LooseVersion
import warnings, sys

try:
  import scipy
  if LooseVersion(scipy.__version__) < LooseVersion('0.15.1'):
    warnings.warn('This recipe has not been tested on scipy version below 0.15.1. It is strongly recommended that an updated scipy version be used.')
    sys.exit(1)
except ImportError:
  warnings.warn('This recipe requires scipy version 0.15.1')
  sys.exit(1)
" || exit 1;
