export KALDI_ROOT=../../..
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sph2pipe_v2.5:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
if [ -f $KALDI_ROOT/tools/env.sh ]; then . $KALDI_ROOT/tools/env.sh; fi
export PATH=$KALDI_ROOT/tools/sctk/bin:$PATH
export LC_ALL=C
