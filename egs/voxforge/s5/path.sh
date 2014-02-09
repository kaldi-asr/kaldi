export KALDI_ROOT=`pwd`/../../..
export PATH=$PWD/utils/:$KALDI_ROOT/src/bin:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/src/fstbin/:$KALDI_ROOT/src/gmmbin/:$KALDI_ROOT/src/featbin/:$KALDI_ROOT/src/lm/:$KALDI_ROOT/src/sgmmbin/:$KALDI_ROOT/src/sgmm2bin/:$KALDI_ROOT/src/fgmmbin/:$KALDI_ROOT/src/latbin/:$PWD:$PATH

# VoxForge data will be stored in:
export DATA_ROOT="/home/dpovey/kaldi-clean/egs/voxforge/s5/voxforge"    # e.g. something like /media/secondary/voxforge

if [ -z $DATA_ROOT ]; then
  echo "You need to set \"DATA_ROOT\" variable in path.sh to point to the directory to host VoxForge's data"
  exit 1
fi

# Needed for "correct" sorting
export LC_ALL=C
