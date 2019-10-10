export KALDI_ROOT=`pwd`/../../..
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

LMBIN=$KALDI_ROOT/tools/irstlm/bin
SRILM=$KALDI_ROOT/tools/srilm/bin/i686-m64

export PATH=$PATH:$LMBIN:$SRILM

# The following was needed to enable Python 3 and also a version of
# gcc consistent with the latest version of cuda on our system. You
# might have to do something similar if you are still on python 2.7
# and have an older version of gcc and a new version of cuda. 

# export LD_LIBRARY_PATH=/opt/share/Python-3.5.2/x86_64/lib:/speech7/picheny4_nb/testi/927/kaldi/egs/ami/s5c/local/lib64:$LD_LIBRARY_PATH
