export KALDI_ROOT=`pwd`/../../../
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LD_LIBRARY_PATH=/home/dpovey/libs

export SPARROWHAWK_ROOT=$KALDI_ROOT/tools/sparrowhawk
export PATH=$SPARROWHAWK_ROOT/bin:$PATH
export LC_ALL=C
export LANG=C

source ~/anaconda/bin/activate py36
