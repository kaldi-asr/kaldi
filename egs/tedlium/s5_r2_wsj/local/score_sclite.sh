#!/usr/bin/env bash
#
# Copyright Johns Hopkins University (Author: Daniel Povey) 2012,
#           Brno University of Technology (Author: Karel Vesely) 2014,
# Apache 2.0
#

# begin configuration section.
cmd=run.pl
stage=0
decode_mbr=true
beam=7  # speed-up, but may affect MBR confidences.
word_ins_penalty=0.0,0.5,1.0
min_lmwt=7
max_lmwt=17
iter=final
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

# name=`basename $data`; # e.g. eval2000
nj=$(cat $dir/num_jobs)

mkdir -p $dir/scoring/log

if [ -f $dir/../frame_shift ]; then
  frame_shift_opt="--frame-shift=$(cat $dir/../frame_shift)"
  echo "$0: $dir/../frame_shift exists, using $frame_shift_opt"
elif [ -f $dir/../frame_subsampling_factor ]; then
  factor=$(cat $dir/../frame_subsampling_factor) || exit 1
  frame_shift_opt="--frame-shift=0.0$factor"
  echo "$0: $dir/../frame_subsampling_factor exists, using $frame_shift_opt"
fi

if [ $stage -le 0 ]; then
  for wip in $(echo $word_ins_penalty | sed 's/,/ /g'); do
    $cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring/log/get_ctm.LMWT.${wip}.log \
      set -e -o pipefail \; \
      mkdir -p $dir/score_LMWT_${wip}/ '&&' \
      lattice-scale --inv-acoustic-scale=LMWT "ark:gunzip -c $dir/lat.*.gz|" ark:- \| \
      lattice-add-penalty --word-ins-penalty=$wip ark:- ark:- \| \
      lattice-prune --beam=$beam ark:- ark:- \| \
      lattice-align-words --output-error-lats=true --max-expand=10.0 --test=false \
       $lang/phones/word_boundary.int $model ark:- ark:- \| \
      lattice-to-ctm-conf --decode-mbr=$decode_mbr $frame_shift_opt ark:- - \| \
      utils/int2sym.pl -f 5 $lang/words.txt \| \
      utils/convert_ctm.pl $data/segments $data/reco2file_and_channel \| \
      sort -k1,1 -k2,2 -k3,3nb '>' $dir/score_LMWT_${wip}/ctm || exit 1;
  done
fi

if [ $stage -le 1 ]; then
  # Remove some stuff we don't want to score, from the ctm.
  for x in $dir/score_*/ctm; do
    # `-i` is not needed in the following. It is added for robustness in ase this code is copy-pasted
    # into another script that, e.g., uses <UNK> instead of <unk>
    grep -v -w -i '<unk>' <$x > ${x}.filt || exit 1;
  done
fi

# Score the set...
if [ $stage -le 2 ]; then
  for wip in $(echo $word_ins_penalty | sed 's/,/ /g'); do
    $cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring/log/score.LMWT.${wip}.log \
      cp $data/stm $dir/score_LMWT_${wip}/ '&&' \
      $hubscr -p $hubdir -V -l english -h hub5 -g $data/glm -r $dir/score_LMWT_${wip}/stm $dir/score_LMWT_${wip}/ctm.filt || exit 1;
  done
fi

exit 0
