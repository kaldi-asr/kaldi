#!/bin/bash
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
word_ins_penalty=0
min_lmwt=10
max_lmwt=20
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

model=$dir/../final.mdl # assume model one level up from decoding dir.

hubscr=$KALDI_ROOT/tools/sctk-2.4.0/bin/hubscr.pl 
[ ! -f $hubscr ] && echo "Cannot find scoring program at $hubscr" && exit 1;
hubdir=`dirname $hubscr`

for f in $data/stm $data/glm $lang/words.txt $lang/phones/word_boundary.int \
     $model $data/segments $data/reco2file_and_channel $dir/lat.1.gz; do
  [ ! -f $f ] && echo "$0: expecting file $f to exist" && exit 1;
done

#name=`basename $data`; # e.g. eval2000
nj=$(cat $dir/num_jobs)

mkdir -p $dir/scoring/log

if [ $stage -le 0 ]; then
  for LMWT in $(seq $min_lmwt $max_lmwt); do
    # Decode lattices to CTMs
    $cmd JOB=1:$nj $dir/score_$LMWT/log/get_ctm.JOB.log \
      lattice-scale --inv-acoustic-scale=$LMWT "ark:gunzip -c $dir/lat.JOB.gz|" ark:- \| \
      lattice-add-penalty --word-ins-penalty=$word_ins_penalty ark:- ark:- \| \
      lattice-prune --beam=$beam ark:- ark:- \| \
      lattice-align-words-lexicon --output-error-lats=true --max-expand=10.0 --test=false \
       $lang/phones/align_lexicon.int $model ark:- ark:- \| \
      lattice-to-ctm-conf --decode-mbr=$decode_mbr ark:- $dir/score_$LMWT/JOB.ctm || exit 1
    # Merge CTMs, sort
    cat $dir/score_$LMWT/*.ctm | \
      utils/int2sym.pl -f 5 $lang/words.txt | \
      utils/convert_ctm.pl $data/segments $data/reco2file_and_channel | \
      sort -k1,1 -k2,2 -k3,3nb > $dir/score_$LMWT/ctm
    rm $dir/score_$LMWT/*.ctm
  done
fi

if [ $stage -le 1 ]; then
  # Remove some stuff we don't want to score, from the ctm.
  $cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring/log/filter_ctm.LMWT.log \
    cat $dir/score_LMWT/ctm \| \
    grep -v -E '"\[BREATH|NOISE|COUGH|SMACK|UM|UH\]"' \| \
    grep -v -E '"!SIL|\<UNK\>"' \
    '>' $dir/score_LMWT/ctm.filt || exit 1
fi

# Score the set...
if [ $stage -le 2 ]; then  
  $cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring/log/score.LMWT.log \
    cp $data/stm $dir/score_LMWT/ '&&' \
    $hubscr -p $hubdir -V -l english -h hub5 -g $data/glm -r $dir/score_LMWT/stm $dir/score_LMWT/ctm.filt || exit 1;
fi

exit 0
