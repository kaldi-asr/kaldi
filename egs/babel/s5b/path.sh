export KALDI_ROOT=/export/a09/jtrmal/kaldi
. $KALDI_ROOT/tools/env.sh
export SRILM=$KALDI_ROOT/tools/srilm/bin/i686-m64
. /export/babel/data/software/env.sh 
export PATH=$PWD/utils/:$KALDI_ROOT/tools/sph2pipe_v2.5/:$KALDI_ROOT/src/bin:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/src/fstbin/:$KALDI_ROOT/src/gmmbin/:$KALDI_ROOT/src/featbin/:$KALDI_ROOT/src/lm/:$KALDI_ROOT/src/sgmmbin/:$KALDI_ROOT/src/sgmm2bin/:$KALDI_ROOT/src/fgmmbin/:$KALDI_ROOT/src/latbin/:$KALDI_ROOT/src/nnet2bin:$KALDI_ROOT/src/nnetbin:$KALDI_ROOT/src/nnet-cpubin/:$KALDI_ROOT/src/kwsbin:$SRILM:$PWD:$PATH:/opt/nvidia_cuda/cuda-5.0/bin
export LC_ALL=C
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/dpovey/libs

