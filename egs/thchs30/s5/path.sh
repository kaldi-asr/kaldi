#Change the following settings to match your computing env,
#so that appropriate (CPU/GPU) Kaldi commands are visible

export KALDI_ROOT=`pwd`/../../..
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh

if [ -z `command nvidia-smi 2>/dev/null` ];then
#set the default as no cuda
    src=src
    echo "using no-cuda kaldi src: $KALDI_ROOT/$src"
else
#assume cuda version in src_cuda. change to your own location
    src=src_cuda
    echo "using cuda kaldi src: $KALDI_ROOT/$src"
fi

export PATH=$PWD/utils/:$KALDI_ROOT/$src/bin:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/$src/fstbin/:$KALDI_ROOT/$src/gmmbin/:$KALDI_ROOT/$src/featbin/:$KALDI_ROOT/$src/lm/:$KALDI_ROOT/$src/sgmmbin/:$KALDI_ROOT/$src/sgmm2bin/:$KALDI_ROOT/$src/fgmmbin/:$KALDI_ROOT/$src/latbin/:$KALDI_ROOT/$src/nnetbin:$KALDI_ROOT/$src/nnet2bin:$PWD:$PATH

export LC_ALL=C

