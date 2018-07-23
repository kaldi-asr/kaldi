# This contains the locations of the tools and data required for running
# the GlobalPhone experiments.

KALDIROOT=/exports/home/aghoshal/kaldi/trunk
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh

KALDISRC=$KALDIROOT/src
KALDIBIN=$KALDISRC/bin:$KALDISRC/featbin:$KALDISRC/fgmmbin:$KALDISRC/fstbin
KALDIBIN=$KALDIBIN:$KALDISRC/gmmbin:$KALDISRC/latbin:$KALDISRC/nnetbin
KALDIBIN=$KALDIBIN:$KALDISRC/sgmm2bin:$KALDISRC/lmbin

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
SHORTEN=$PWD/tools/shorten-3.6.1/bin
SOX=$PWD/tools/sox-14.3.2/bin
[ -x $SHORTEN/shorten ] || { echo "Cannot find shorten executable"; }
[ -x $SOX/sox ] || { echo "Cannot find sox executable"; }
TOOLS=$SHORTEN:$SOX

export PATH=$PATH:$KALDIBIN:$FSTBIN:$LMBIN:$SCRIPTS:$TOOLS
export LC_ALL=C

# Site-specific configs:
[ `hostname -y` == ecdf ] && { . path_ed.sh; }
