export KALDI_ROOT=`pwd`/../../..
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/irstlm/bin/:$KALDI_ROOT/tools/kaldi_lm/:$PWD:$PATH
[ ! -f $KALDI_ROOT/src/path.sh ] && echo >&2 "The standard file $KALDI_ROOT/src/path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/src/path.sh
export LC_ALL=C

BEAMFORMIT=$KALDI_ROOT/tools/BeamformIt-3.51

export PATH=$PATH:$BEAMFORMIT
