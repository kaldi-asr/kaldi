# Defining Kaldi root directory
export KALDI_ROOT=`pwd`/../..
 
# Setting paths to useful tools
export PATH=$PWD/utils/:$KALDI_ROOT/src/bin:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/src/fstbin/:$KALDI_ROOT/src/gmmbin/:$KALDI_ROOT/src/featbin/:$KALDI_ROOT/src/lmbin/:$KALDI_ROOT/src/sgmm2bin/:$KALDI_ROOT/src/fgmmbin/:$KALDI_ROOT/src/latbin/:$KALDI_ROOT/tools/srilm/bin/:$PWD:$PATH
 
# Defining audio data directory (modify it for your installation directory!)

export DATA_ROOT="/home/abhilash/kaldi/egs/0_MP_Digits/digits_audio"
 
# Enable SRILM
source $KALDI_ROOT/tools/env.sh
 
# Variable needed for proper data sorting
export LC_ALL=C
