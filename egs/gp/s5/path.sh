# This contains the locations of the tools and data required for running
# the GlobalPhone experiments.

KALDI_ROOT=/homes/eva/q/qghoshal/src/kaldi/trunk
KALDISRC=$KALDI_ROOT/src
KALDIBIN=$KALDISRC/bin:$KALDISRC/featbin:$KALDISRC/fgmmbin:$KALDISRC/fstbin  
KALDIBIN=$KALDIBIN:$KALDISRC/gmmbin:$KALDISRC/latbin:$KALDISRC/nnetbin
KALDIBIN=$KALDIBIN:$KALDISRC/sgmmbin:$KALDISRC/tiedbin:$KALDISRC/lm

FSTBIN=$KALDI_ROOT/tools/openfst/bin
LMBIN=$KALDI_ROOT/tools/irstlm/bin

[ -d $PWD/local ] || { echo "Error: 'local' subdirectory not found."; }
[ -d $PWD/utils ] || { echo "Error: 'utils' subdirectory not found."; }
[ -d $PWD/steps ] || { echo "Error: 'steps' subdirectory not found."; }

export kaldi_local=$PWD/local
export kaldi_utils=$PWD/utils
export kaldi_steps=$PWD/steps
SCRIPTS=$kaldi_local:$kaldi_utils:$kaldi_steps

# # If you already have shorten and sox on your path, comment the following out.
# # Else use install.sh to install them first in the specified locations.
# SHORTEN=$PWD/tools/shorten-3.6.1/bin
# SOX=$PWD/tools/sox-14.3.2/bin
# [ -x $SHORTEN/shorten ] || { echo "Cannot find shorten executable"; }
# [ -x $SOX/sox ] || { echo "Cannot find sox executable"; }
# TOOLS=$SHORTEN:$SOX

export PATH=$PATH:$KALDIBIN:$FSTBIN:$LMBIN:$SCRIPTS
#export PATH=$PATH:$KALDIBIN:$FSTBIN:$LMBIN:$SCRIPTS:$TOOLS
export LC_COLLATE=C  # For expected sorting behaviour
