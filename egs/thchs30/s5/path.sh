#Change the following settings to match your computing env,
#so that appropriate (CPU/GPU) Kaldi commands are visible

export KALDI_ROOT=/work3/zxw/kaldi-2016-1-21

if [ -z `which nvidia-smi 2>/dev/null` ];then
#if nvidia_smi; then
    src=src_nocuda
    echo "using no-cuda kaldi src: $KALDI_ROOT src"
else
    src=src_cuda
    echo "using cuda kaldi src: $KALDI_ROOT src"
fi

export PATH=$PWD/utils/:$KALDI_ROOT/$src/nnet:$KALDI_ROOT/$src/nnetbin:$KALDI_ROOT/$src/bin:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/$src/fstbin/:$KALDI_ROOT/$src/gmmbin/:$KALDI_ROOT/$src/featbin/:$KALDI_ROOT/$src/lm/:$KALDI_ROOT/$src/sgmmbin/:$KALDI_ROOT/$src/sgmm2bin/:$KALDI_ROOT/$src/fgmmbin/:$KALDI_ROOT/$src/latbin/:$KALDI_ROOT/$src/nnet-cpubin/:$KALDI_ROOT/$src/kwsbin:$KALDI_ROOT/$src/nnet2bin:$PATH

export LC_ALL=C

