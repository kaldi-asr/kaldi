#!/bin/bash
# Copyright 2012  Johns Hopkins University (Author: Daniel Povey)
#           2014  Brno University of Technology (Author: Karel Vesely)
# Apache 2.0

# begin configuration section.
cmd=run.pl
stage=0
min_lmwt=1
max_lmwt=10
mbr_scale=1.0
#end configuration section.

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
  echo "Usage: local/score.sh [--cmd (run.pl|queue.pl...)] <data-dir> <lang-dir|graph-dir> <decode-dir>"
  echo " Options:"
  echo "    --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes."
  echo "    --stage (0|1|2)                 # start scoring script from part-way through."
  echo "    --min_lmwt <int>                # minumum LM-weight for lattice rescoring "
  echo "    --max_lmwt <int>                # maximum LM-weight for lattice rescoring "
  exit 1;
fi

data=$1
lang_or_graph=$2
dir=$3

model=$dir/../final.mdl # assume model one level up from decoding dir.

hubscr=$KALDI_ROOT/tools/sctk/bin/hubscr.pl 
[ ! -f $hubscr ] && echo "Cannot find scoring program at $hubscr" && exit 1;
hubdir=`dirname $hubscr`

phonemap="conf/phones.60-48-39.map"
nj=$(cat $dir/num_jobs)

symtab=$lang_or_graph/words.txt

for f in $symtab $dir/lat.1.gz $data/text; do
  [ ! -f $f ] && echo "score.sh: no such file $f" && exit 1;
done

mkdir -p $dir/scoring/log

# Map reference to 39 phone classes, the silence is optional (.):
local/timit_norm_trans.pl -i $data/stm -m $phonemap -from 48 -to 39 >$dir/scoring/stm_39phn
cp $data/glm $dir/scoring/glm_39phn

if [ $stage -le 0 ]; then
  # Get the phone-sequence on the best-path:
  for LMWT in $(seq $min_lmwt $max_lmwt); do
    $cmd JOB=1:$nj $dir/scoring/log/best_path.$LMWT.JOB.log \
      lattice-align-phones $model "ark:gunzip -c $dir/lat.JOB.gz|" ark:- \| \
      lattice-to-ctm-conf --acoustic-scale=$(bc <<<"scale=8; 1/$LMWT*$mbr_scale") --lm-scale=$mbr_scale ark:- $dir/scoring/$LMWT.JOB.ctm || exit 1;
    cat $dir/scoring/$LMWT.*.ctm | sort > $dir/scoring/$LMWT.ctm
    rm $dir/scoring/$LMWT.*.ctm
  done
fi

if [ $stage -le 1 ]; then
  # Map ctm to 39 phone classes:
  $cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring/log/map_ctm.LMWT.log \
     mkdir $dir/score_LMWT ';' \
     cat $dir/scoring/LMWT.ctm \| \
     utils/int2sym.pl -f 5 $symtab \| \
     local/timit_norm_trans.pl -i - -m $phonemap -from 48 -to 39 '>' \
     $dir/scoring/LMWT.ctm_39phn || exit 1
fi

# Score the set...
$cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring/log/score.LMWT.log \
  cp $dir/scoring/stm_39phn $dir/score_LMWT/stm_39phn '&&' cp $dir/scoring/LMWT.ctm_39phn $dir/score_LMWT/ctm_39phn '&&' \
   $hubscr -p $hubdir -V -l english -h hub5 -g $dir/scoring/glm_39phn -r $dir/score_LMWT/stm_39phn $dir/score_LMWT/ctm_39phn || exit 1;

exit 0;
