export KALDI_ROOT=`pwd`/../../../
export PWD=`pwd`
export PATH=$KALDI_ROOT/src/ivectorbin:$PWD/stanford-utils:$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$KALDI_ROOT/tools/kaldi_lm:$KALDI_ROOT/tools/srilm/bin:$KALDI_ROOT/tools/srilm/bin/i686-m64:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C
