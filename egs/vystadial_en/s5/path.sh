export KALDI_ROOT=`pwd`/../../..
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/irstlm/bin/:$PWD:$PWD/utils:$PWD/steps:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C



srilm_bin=$KALDI_ROOT/tools/srilm/bin/
if [ ! -e "$srilm_bin" ] ; then
    echo "SRILM is not installed in $KALDI_ROOT/tools."
    echo "May not be able to create LMs!"
    echo "Please go to $KALDI_ROOT/tools and run ./extras/install_srilm.sh"
fi
srilm_sub_bin=`find "$srilm_bin" -type d`
for d in $srilm_sub_bin ; do
    export PATH=$d:$PATH
done
