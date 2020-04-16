#!/usr/bin/env bash
# Copyright Johns Hopkins University (Author: Daniel Povey) 2012.  Apache 2.0.

# begin configuration section.
cmd=run.pl
stage=0
min_lmwt=5
max_lmwt=17
iter=final
word_ins_penalty=0.0,0.5,1.0
resolve_ctm_overlaps=false
#end configuration section.

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
  echo "Usage: local/score_sclite.sh [--cmd (run.pl|queue.pl...)] <data-dir> <lang-dir|graph-dir> <decode-dir>"
  echo " Options:"
  echo "    --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes."
  echo "    --stage (0|1|2)                 # start scoring script from part-way through."
  echo "    --min_lmwt <int>                # minumum LM-weight for lattice rescoring "
  echo "    --max_lmwt <int>                # maximum LM-weight for lattice rescoring "
  exit 1;
fi

data=$1
lang=$2 # Note: may be graph directory not lang directory, but has the necessary stuff copied.
dir=$3

model=$dir/../$iter.mdl # assume model one level up from decoding dir.

hubscr=$KALDI_ROOT/tools/sctk/bin/hubscr.pl
[ ! -f $hubscr ] && echo "Cannot find scoring program at $hubscr" && exit 1;
hubdir=`dirname $hubscr`

for f in $data/stm $data/glm $lang/words.txt $lang/phones/word_boundary.int \
     $model $data/segments $data/reco2file_and_channel $dir/lat.1.gz; do
  [ ! -f $f ] && echo "$0: expecting file $f to exist" && exit 1;
done

if [ -f $dir/../frame_shift ]; then
  frame_shift_opt="--frame-shift=$(cat $dir/../frame_shift)"
  echo "$0: $dir/../frame_shift exists, using $frame_shift_opt"
elif [ -f $dir/../frame_subsampling_factor ]; then
  factor=$(cat $dir/../frame_subsampling_factor) || exit 1
  frame_shift_opt="--frame-shift=0.0$factor"
  echo "$0: $dir/../frame_subsampling_factor exists, using $frame_shift_opt"
fi

name=`basename $data`; # e.g. eval2000

mkdir -p $dir/scoring/log

if [ $stage -le 0 ]; then
  for wip in $(echo $word_ins_penalty | sed 's/,/ /g'); do
    $cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring/log/get_ctm.LMWT.${wip}.log \
      mkdir -p $dir/score_LMWT_${wip}/ '&&' \
      lattice-scale --lm-scale=LMWT "ark:gunzip -c $dir/lat.*.gz|" ark:- \| \
      lattice-add-penalty --word-ins-penalty=$wip ark:- ark:- \| \
      lattice-1best ark:- ark:- \| \
      lattice-align-words $lang/phones/word_boundary.int $model ark:- ark:- \| \
      nbest-to-ctm $frame_shift_opt ark:- - \| \
      utils/int2sym.pl -f 5 $lang/words.txt '>' \
      $dir/score_LMWT_${wip}/$name.utt_ctm || exit 1;
  done
fi

if [ $stage -le 1 ]; then
  for wip in $(echo $word_ins_penalty | sed 's/,/ /g'); do
    if $resolve_ctm_overlaps; then
      $cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring/log/resolve_ctm_overlaps.LMWT.${wip}.log \
        utils/ctm/resolve_ctm_overlaps.py $data/segments \
          $dir/score_LMWT_${wip}/$name.utt_ctm - \| \
        utils/convert_ctm.pl $data/segments $data/reco2file_and_channel \
        '>' $dir/score_LMWT_${wip}/$name.ctm || exit 1;
    else
      $cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring/log/convert_ctm.LMWT.${wip}.log \
        cat $dir/score_LMWT_${wip}/$name.utt_ctm \| \
        utils/convert_ctm.pl $data/segments $data/reco2file_and_channel \
        '>' $dir/score_LMWT_${wip}/$name.ctm || exit 1;
    fi
  done
fi

if [ $stage -le 2 ]; then
  # Remove some stuff we don't want to score, from the ctm.
  # the big expression in parentheses contains all the things that get mapped
  # by the glm file, into hesitations.
  # The -$ expression removes partial words.
  # the aim here is to remove all the things that appear in the reference as optionally
  # deletable (inside parentheses), as if we delete these there is no loss, while
  # if we get them correct there is no gain.
  for x in $dir/score_*/$name.ctm; do
    cp $x $dir/tmpf;
    cat $dir/tmpf | grep -i -v -E '<NOISE|SPOKEN_NOISE>' | \
    grep -i -v -E ' (UH|UM|EH|MM|HM|AH|HUH|HA|ER|OOF|HEE|ACH|EEE|EW)$' | \
    grep -v -- '-$' > $x;
  done
fi

# Score the set...
if [ $stage -le 3 ]; then
  for wip in $(echo $word_ins_penalty | sed 's/,/ /g'); do
    $cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring/log/score.LMWT.${wip}.log \
      cp $data/stm $dir/score_LMWT_${wip}/ '&&' \
      $hubscr -p $hubdir -V -l english -h hub4 -g $data/glm -r $dir/score_LMWT_${wip}/stm $dir/score_LMWT_${wip}/${name}.ctm || exit 1;
  done
fi

exit 0
