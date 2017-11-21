export KALDI_ROOT=/home/vmanoha1/kaldi-diarization-v2
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sph2pipe_v2.5:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export PATH=/home/vmanoha1/kaldi-diarization-v2/src/ivectorbin/:$PATH
export PATH=/home/vmanoha1/kaldi-diarization-v2/src/segmenterbin/:$PATH
export PATH=$KALDI_ROOT/tools/sctk/bin:$PATH
export LC_ALL=C
