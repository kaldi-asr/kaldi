export KALDI_ROOT=`pwd`/../../..
. /export/babel/data/software/env.sh 
. $KALDI_ROOT/tools/config/common_path.sh
export PATH=$PWD/utils/:$KALDI_ROOT/tools/sph2pipe_v2.5/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH
. $KALDI_ROOT/tools/env.sh
export LC_ALL=C

