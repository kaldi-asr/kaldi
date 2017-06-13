#!/bin/bash

# Script to check the tool versions necessary for the aspire recipe
function check_for_bad_sox {
  if which sox >&/dev/null; then  # sox is on the path
    sox_version=$(sox --version | awk -F 'v' '{print $2}' | awk -F '.' '{print $1 "." $2}')
    if [ "$sox_version" == "14.2" ] || [ "$sox_version" == "14.3" ]; then
      echo "*** WARNING: your version of sox is either 14.2.x or 14.3.x ***"
      echo "*** which may cause errors in the data preparation of this recipe. ***"
      echo "*** Please upgrade your sox to version 14.4 or higher. ***"
      exit 1;
    fi
  else
    echo "*** This recipe requires sox for the data preparation. ***"
    exit 1;
  fi
}

check_for_bad_sox;

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
