#!/usr/bin/env bash
# Copyright 2012  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0

[ -f ./path.sh ] && . ./path.sh

# begin configuration section.
cmd=run.pl
stage=0
decode_mbr=true
reverse=false
word_ins_penalty=0.0
min_lmwt=9
max_lmwt=30
iter=final

#end configuration section.

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
  echo "Usage: local/score.sh [--cmd (run.pl|queue.pl...)] <data-dir> <lang-dir|graph-dir> <decode-dir>"
  echo " Options:"
  echo "    --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes."
  echo "    --stage (0|1|2)                 # start scoring script from part-way through."
  echo "    --decode_mbr (true/false)       # maximum bayes risk decoding (confusion network)."
  echo "    --min_lmwt <int>                # minumum LM-weight for lattice rescoring "
  echo "    --max_lmwt <int>                # maximum LM-weight for lattice rescoring "
  echo "    --reverse (true/false)          # score with time reversed features "
  exit 1;
fi

data=$1
lang_or_graph=$2
dir=$3
srcdir=`dirname $dir`;
symtab=$lang_or_graph/words.txt

for f in $symtab $dir/lat.1.gz $data/text; do
  [ ! -f $f ] && echo "score.sh: no such file $f" && exit 1;
done

name=`basename $data`

mkdir -p $dir/scoring/log

if [ -f $dir/../frame_shift ]; then
  frame_shift_opt="--frame-shift=$(cat $dir/../frame_shift)"
  echo "$0: $dir/../frame_shift exists, using $frame_shift_opt"
elif [ -f $dir/../frame_subsampling_factor ]; then
  factor=$(cat $dir/../frame_subsampling_factor) || exit 1
  frame_shift_opt="--frame-shift=0.0$factor"
  echo "$0: $dir/../frame_subsampling_factor exists, using $frame_shift_opt"
fi

sclite=$KALDI_ROOT/tools/sctk/bin/sclite
[ ! -f $sclite ] && echo "Cannot find scoring program at $sclite" && exit 1

#cat $data/text | sed 's:<NOISE>::g' | sed 's:<SPOKEN_NOISE>::g' > $dir/scoring/test_filt.txt
if [ $stage -le 0 ]; then
  $cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring/log/best_path.LMWT.log \
    mkdir -p $dir/score_LMWT/ '&&' \
    lattice-1best --lm-scale=LMWT "ark:gunzip -c $dir/lat.*.gz|" ark:- \| \
    lattice-align-words $lang_or_graph/phones/word_boundary.int $srcdir/$iter.mdl ark:- ark:- \| \
    nbest-to-ctm $frame_shift_opt ark:- - \| \
    utils/int2sym.pl -f 5 $symtab  \| \
    utils/convert_ctm.pl $data/segments $data/reco2file_and_channel \
    '>' $dir/score_LMWT/${name}.ctm '&&' \
    grep -v '\<UNK\>' $dir/score_LMWT/${name}.ctm \| \
    sed -e 's:^[^ ]*\/::' -e 's:.wav::' \| sort -k1,1 -k3,3n \
    '>' $dir/score_LMWT/${name}.ctm.updated || exit 1;
fi    

# Remove some stuff we don't want to score, from the ctm.                                                                                                   

if [ $stage -le 1 ]; then 
  for x in $dir/score_*/$name.ctm.updated; do
    cp $x $dir/tmpf;
    cat $dir/tmpf | grep -v -E '\[NOISE|LAUGHTER|VOCALIZED-NOISE\]' | \
      grep -v -E '<UNK>|%HESITATION' > $x;
  done
fi  

if [ $stage -le 2 ]; then
  $cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring/log/score.LMWT.log \
    local/toutf8.py $dir/score_LMWT/${name}.ctm.updated $dir/score_LMWT/${name}.ctm.updated.utf8 '&&' \
    local/toutf8.py $data/stm $dir/score_LMWT/stm '&&' \
    $sclite -O $dir/score_LMWT -o all spk -h $dir/score_LMWT/${name}.ctm.updated.utf8 ctm -r $dir/score_LMWT/stm stm || exit 1;  
fi 

