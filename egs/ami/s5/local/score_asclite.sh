#!/bin/bash
# Copyright Johns Hopkins University (Author: Daniel Povey) 2012.  Apache 2.0.
# 2014, University of Edinburgh, (Author: Pawel Swietojanski)

# begin configuration section.
cmd=run.pl
stage=0
min_lmwt=9
max_lmwt=20
reverse=false
asclite=true
overlap_spk=4
#end configuration section.

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
  echo "Usage: local/score_asclite.sh [--cmd (run.pl|queue.pl...)] <data-dir> <lang-dir|graph-dir> <decode-dir>"
  echo " Options:"
  echo "    --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes."
  echo "    --stage (0|1|2)                 # start scoring script from part-way through."
  echo "    --min_lmwt <int>                # minumum LM-weight for lattice rescoring "
  echo "    --max_lmwt <int>                # maximum LM-weight for lattice rescoring "
  echo "    --reverse (true/false)          # score with time reversed features "
  exit 1;
fi

data=$1
lang=$2 # Note: may be graph directory not lang directory, but has the necessary stuff copied.
dir=$3

model=$dir/../final.mdl # assume model one level up from decoding dir.

hubscr=$KALDI_ROOT/tools/sctk/bin/hubscr.pl 
[ ! -f $hubscr ] && echo "Cannot find scoring program at $hubscr" && exit 1;
hubdir=`dirname $hubscr`

for f in $data/stm $data/glm $lang/words.txt $lang/phones/word_boundary.int \
     $model $data/segments $data/reco2file_and_channel $dir/lat.1.gz; do
  [ ! -f $f ] && echo "$0: expecting file $f to exist" && exit 1;
done

name=`basename $data`; # e.g. eval2000

mkdir -p $dir/ascoring/log

if [ $stage -le 0 ]; then
  if $reverse; then
    $cmd LMWT=$min_lmwt:$max_lmwt $dir/ascoring/log/get_ctm.LMWT.log \
      mkdir -p $dir/ascore_LMWT/ '&&' \
      lattice-1best --lm-scale=LMWT "ark:gunzip -c $dir/lat.*.gz|" ark:- \| \
      lattice-reverse ark:- ark:- \| \
      lattice-align-words --reorder=false $lang/phones/word_boundary.int $model ark:- ark:- \| \
      nbest-to-ctm ark:- - \| \
      utils/int2sym.pl -f 5 $lang/words.txt  \| \
      utils/convert_ctm.pl $data/segments $data/reco2file_and_channel \
      '>' $dir/ascore_LMWT/$name.ctm || exit 1;
  else
    $cmd LMWT=$min_lmwt:$max_lmwt $dir/ascoring/log/get_ctm.LMWT.log \
      mkdir -p $dir/ascore_LMWT/ '&&' \
      lattice-1best --lm-scale=LMWT "ark:gunzip -c $dir/lat.*.gz|" ark:- \| \
      lattice-align-words $lang/phones/word_boundary.int $model ark:- ark:- \| \
      nbest-to-ctm ark:- - \| \
      utils/int2sym.pl -f 5 $lang/words.txt  \| \
      utils/convert_ctm.pl $data/segments $data/reco2file_and_channel \
      '>' $dir/ascore_LMWT/$name.ctm || exit 1;
  fi
fi

if [ $stage -le 1 ]; then
# Remove some stuff we don't want to score, from the ctm.
  for x in $dir/ascore_*/$name.ctm; do
    cp $x $dir/tmpf;
    cat $dir/tmpf | grep -i -v -E '\[noise|laughter|vocalized-noise\]' | \
      grep -i -v -E '<unk>' > $x;
#      grep -i -v -E '<UNK>|%HESITATION' > $x;
  done
fi

if [ $stage -le 2 ]; then  
  if [ "$asclite" == "true" ]; then
    oname=$name
    [ ! -z $overlap_spk ] && oname=${name}_o$overlap_spk
    $cmd LMWT=$min_lmwt:$max_lmwt $dir/ascoring/log/score.LMWT.log \
      cp $data/stm $dir/ascore_LMWT/ '&&' \
      cp $dir/ascore_LMWT/${name}.ctm $dir/ascore_LMWT/${oname}.ctm '&&' \
      $hubscr -G -v -m 1:2 -o$overlap_spk -a -C -B 8192 -p $hubdir -V -l english \
         -h rt-stt -g $data/glm -r $dir/ascore_LMWT/stm $dir/ascore_LMWT/${oname}.ctm || exit 1;
  else
    $cmd LMWT=$min_lmwt:$max_lmwt $dir/ascoring/log/score.LMWT.log \
      cp $data/stm $dir/ascore_LMWT/ '&&' \
      $hubscr -p $hubdir -V -l english -h hub5 -g $data/glm -r $dir/ascore_LMWT/stm $dir/ascore_LMWT/${name}.ctm || exit 1
  fi
fi

exit 0
