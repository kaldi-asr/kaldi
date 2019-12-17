export KALDI_ROOT=`pwd`/../../..
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sph2pipe_v2.5/:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
[ ! -f $KALDI_ROOT/tools/env.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/env.sh is not present (this is uncommon but might be OK)"
. $KALDI_ROOT/tools/env.sh
export LC_ALL=C
