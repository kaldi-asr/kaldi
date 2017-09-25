export KALDI_ROOT=`pwd`/../../..
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export PYTHONPATH=${PYTHONPATH:+$PYTHONPATH:}$KALDI_ROOT/tools/tensorflow_build/.local/lib/python2.7/site-packages
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$KALDI_ROOT/tools/tensorflow/bazel-bin/tensorflow/:/usr/local/cuda/lib64:/export/a11/hlyu/cudnn/lib64:/home/dpovey/libs/
export LC_ALL=C
