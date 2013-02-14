# This contains the locations of the tools and data required for running
# the TIMIT experiments.

# The KALDIROOT enviromnent variable must be set by the user.
# KALDIROOT=/absolute/path/to/kaldi/installation
KALDISRC=$KALDIROOT/src
KALDIBIN=$KALDISRC/bin:$KALDISRC/featbin:$KALDISRC/fgmmbin:$KALDISRC/fstbin  
KALDIBIN=$KALDIBIN:$KALDISRC/gmmbin:$KALDISRC/latbin:$KALDISRC/nnetbin
KALDIBIN=$KALDIBIN:$KALDISRC/sgmmbin:$KALDISRC/tiedbin:$KALDISRC/lm

FSTBIN=$KALDIROOT/tools/openfst/bin
LMBIN=$KALDIROOT/tools/irstlm/bin

[ -d $PWD/local ] || { echo "Expecting 'local' subdirectory"; exit 1; }
[ -d $PWD/utils ] || { echo "Expecting 'utils' subdirectory"; exit 1; }
[ -d $PWD/steps ] || { echo "Expecting 'steps' subdirectory"; exit 1; }

LOCALUTILS=$PWD/local
KALDIUTILS=$PWD/utils
KALDISTEPS=$PWD/steps
SCRIPTS=$LOCALUTILS:$KALDIUTILS:$KALDISTEPS

# If you already have shorten and sox on your path, comment the following out.
# Else use install.sh to install them first in the specified locations.
SPH2PIPE=$KALDIROOT/tools/sph2pipe_v2.5
[ -x $SPH2PIPE/sph2pipe ] || { echo "Cannot find sph2pipe executable"; }
TOOLS=$SPH2PIPE

export PATH=$PATH:$KALDIBIN:$FSTBIN:$LMBIN:$SCRIPTS:$TOOLS
export LC_ALL=C
export IRSTLM=$KALDIROOT/tools/irstlm

## Site-specific configs for Edinburgh
# [ `hostname -y` == ecdf ] && \
#   { . /etc/profile.d/modules.sh; module add intel/mkl; }
