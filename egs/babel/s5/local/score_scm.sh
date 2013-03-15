#!/bin/bash
# Copyright Johns Hopkins University (Author: Daniel Povey) 2012.  Apache 2.0.

# begin configuration section.
cmd=run.pl
cer=0
decode_mbr=true
min_lmwt=7
max_lmwt=17
model=
stage=0
#end configuration section.

echo $0 $@

[ -f ./path.sh ] && . ./path.sh
[ -f ./cmd.sh ]  && . ./cmd.sh
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
  echo "Usage: $0 [options] <dataDir> <langDir|graphDir> <decodeDir>" && exit;
  echo " Options:"
  echo "    --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes."
  echo "    --cer (0|1)                     # compute CER in addition to WER"
  exit 1;
fi

data=$1
lang=$2 # This parameter is not used -- kept only for backwards compatibility
dir=$3



ScoringProgram=`which sclite` || ScoringProgram=$KALDI_ROOT/tools/sctk-2.4.0/bin/sclite
[ ! -x $ScoringProgram ] && echo "Cannot find scoring program at $ScoringProgram" && exit 1;


for f in $data/stm $data/glm ; do
  [ ! -f $f ] && echo "$0: expecting file $f to exist" && exit 1;
done

if [ $cer -eq 1 ] ; then
  for f in $data/char.stm  ; do
    [ ! -f $f ] && echo "$0: expecting file $f to exist" && exit 1;
  done
fi


name=`basename $data`; # e.g. eval2000

mkdir -p $dir/scoring/log
if [ $stage -le 0 ] ; then
  $cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring/log/score.LMWT.log \
    cp $data/stm $dir/score_LMWT/ '&&' cp $data/glm $dir/score_LMWT/ '&&'\
    $ScoringProgram -s -r $data/stm stm -h $dir/score_LMWT/${name}.ctm ctm -o all -o dtl || exit 1
fi

if [ $stage -le 1 ]; then
  if [ $cer -eq 1 ]; then
    $cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring/log/score.LMWT.char.log \
      cp $data/char.stm $dir/score_LMWT/'&&'\
      $ScoringProgram -s -r $dir/score_LMWT/char.stm stm -h $dir/score_LMWT/${name}.char.ctm ctm -o all -o dtl || exit 1
  fi
fi


echo "Finished scoring on" `date`
exit 0

