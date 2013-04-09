#!/bin/bash
#
# This is for when you already have the CTMS in <decode-dir>/score_<LMWT>/${name}.ctm
# and just want to do the scoring part.

# begin configuration section.
cmd=run.pl
cer=0
decode_mbr=true
min_lmwt=7
max_lmwt=17
model=
stage=0
ctm_name=
case_insensitive=true
use_icu=true
icu_transform='Any-Lower'
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
SortingProgram=`which hubscr.pl` || ScoringProgram=$KALDI_ROOT/tools/sctk-2.4.0/bin/hubscr.pl
[ ! -x $ScoringProgram ] && echo "Cannot find scoring program at $ScoringProgram" && exit 1;


for f in $data/stm $data/glm ; do
  [ ! -f $f ] && echo "$0: expecting file $f to exist" && exit 1;
done

if [ $cer -eq 1 ] ; then
  for f in $data/char.stm  ; do
    [ ! -f $f ] && echo "$0: expecting file $f to exist" && exit 1;
  done
fi

if [ -z $ctm_name ] ; then
  name=`basename $data`; # e.g. eval2000
else
  name=$ctm_name
fi

mkdir -p $dir/scoring/log
if [ $stage -le 0 ] ; then
  $cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring/log/score.LMWT.log \
    cp $data/stm $dir/score_LMWT/stm.unsorted '&&' \
    cp $data/glm $dir/score_LMWT/glm '&&' \
    cp $dir/score_LMWT/${name}.ctm $dir/score_LMWT/${name}.ctm.unsorted '&&'\
    $SortingProgram sortSTM \<$dir/score_LMWT/stm.unsorted          \>$dir/score_LMWT/stm.sorted '&&' \
    utils/fix_ctm.sh $dir/score_LMWT/stm.sorted $dir/score_LMWT/${name}.ctm.unsorted '&&' \
    $SortingProgram sortCTM \<$dir/score_LMWT/${name}.ctm.unsorted  \>$dir/score_LMWT/${name}.ctm.sorted '&&' \
    paste -d ' ' \<\(cut -f 1-5 -d ' ' $dir/score_LMWT/stm.sorted \) \
                 \<\(cut -f 6- -d ' ' $dir/score_LMWT/stm.sorted \| uconv -f utf8 -t utf8 -x "$icu_transform" \) \
        \> $dir/score_LMWT/stm '&&' \
    paste -d ' ' \<\(cut -f 1-4 -d ' ' $dir/score_LMWT/${name}.ctm.sorted \) \
                 \<\(cut -f 5-  -d ' ' $dir/score_LMWT/${name}.ctm.sorted \| uconv -f utf8 -t utf8 -x "$icu_transform" \) \
        \> $dir/score_LMWT/${name}.ctm '&&' \
    $ScoringProgram -s -r $dir/score_LMWT/stm  stm -h $dir/score_LMWT/${name}.ctm ctm \
      -n "$name.ctm" -f 0 -D -F  -o  sum rsum prf dtl sgml -e utf-8 || exit 1
fi

if [ $stage -le 1 ]; then
  if [ $cer -eq 1 ]; then
    $cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring/log/score.LMWT.char.log \
      $ScoringProgram -s -r $dir/score_LMWT/stm stm -h $dir/score_LMWT/${name}.ctm ctm \
        -n "$name.char.ctm" -o sum rsum prf dtl sgml -f 0 -D -F -c NOASCII DH -e utf-8 || exit 1
  fi
fi


echo "Finished scoring on" `date`
exit 0

