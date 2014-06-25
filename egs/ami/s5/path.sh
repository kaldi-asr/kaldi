
export LC_ALL=C  # For expected sorting and joining behaviour

KALDI_ROOT=/gpfs/scratch/s1136550/kaldi-code

KALDISRC=$KALDI_ROOT/src
KALDIBIN=$KALDISRC/bin:$KALDISRC/featbin:$KALDISRC/fgmmbin:$KALDISRC/fstbin  
KALDIBIN=$KALDIBIN:$KALDISRC/gmmbin:$KALDISRC/latbin:$KALDISRC/nnetbin
KALDIBIN=$KALDIBIN:$KALDISRC/sgmmbin:$KALDISRC/tiedbin

FSTBIN=$KALDI_ROOT/tools/openfst/bin
LMBIN=$KALDI_ROOT/tools/irstlm/bin
BEAMFORMIT=$KALDI_ROOT/tools/BeamformIt-3.51

[ -d $PWD/local ] || { echo "Error: 'local' subdirectory not found."; }
[ -d $PWD/utils ] || { echo "Error: 'utils' subdirectory not found."; }
[ -d $PWD/steps ] || { echo "Error: 'steps' subdirectory not found."; }

export kaldi_local=$PWD/local
export kaldi_utils=$PWD/utils
export kaldi_steps=$PWD/steps
SCRIPTS=$kaldi_local:$kaldi_utils:$kaldi_steps

PATH=$PATH:$KALDIBIN:$FSTBIN:$LMBIN:$SCRIPTS:$BEAMFORMIT

#CUDA_VER='cuda-5.0.35'

#export PATH=$PATH:/opt/$CUDA_VER/bin
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/$CUDA_VER/lib64:/opt/$CUDA_VER/lib



