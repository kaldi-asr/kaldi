#!/bin/bash
# Copyright 2012  Johns Hopkins University (Author: Daniel Povey, Yenda Trmal)
# Apache 2.0

[ -f ./path.sh ] && . ./path.sh

# begin configuration section.
cmd=run.pl
stage=0
decode_mbr=false
reverse=false
stats=true
beam=6
word_ins_penalty=1.5,1.0,0.5,0.0,-0.5,-1.5,-2.0,-2.5,-3.0,-3.5
min_lmwt=9
max_lmwt=20
#end configuration section.

echo "$0 $@"  # Print the command line for logging
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

symtab=$lang_or_graph/words.txt

for f in $symtab $dir/lat.1.gz $data/text; do
  [ ! -f $f ] && echo "score.sh: no such file $f" && exit 1;
done


filtering_cmd="cat -"
[ -x local/wer_output_filter ] && filtering_cmd="local/wer_output_filter"
[ -x local/wer_ref_filter ] && filtering_cmd="local/wer_ref_filter"
if $decode_mbr ; then
  echo "$0: scoring with MBR, WIP=$word_ins_penalty"
else
  echo "$0: scoring with WIP=$word_ins_penalty"
fi


for wip in `echo $word_ins_penalty | sed 's/,/ /g'` ; do
  mkdir -p $dir/scoring_$wip/log
  cat $data/text | $filtering_cmd > $dir/scoring_$wip/test_filt.txt

  if $decode_mbr ; then
    $cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring_$wip/log/best_path.LMWT.log \
      acwt=\`perl -e \"print 1.0/LMWT\"\`\; \
      lattice-scale --inv-acoustic-scale=LMWT "ark:gunzip -c $dir/lat.*.gz|" ark:- \| \
      lattice-add-penalty --word-ins-penalty=$wip ark:- ark:- \| \
      lattice-prune --beam=$beam ark:- ark:- \| \
      lattice-mbr-decode  --word-symbol-table=$symtab \
        ark:- ark,t:$dir/scoring_$wip/LMWT.tra || exit 1;
  else
    $cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring_$wip/log/best_path.LMWT.log \
      lattice-scale --inv-acoustic-scale=LMWT "ark:gunzip -c $dir/lat.*.gz|" ark:- \| \
      lattice-add-penalty --word-ins-penalty=$wip ark:- ark:- \| \
      lattice-best-path --word-symbol-table=$symtab \
        ark:- ark,t:$dir/scoring_$wip/LMWT.tra || exit 1;
  fi
  if $reverse; then
    for lmwt in `seq $min_lmwt $max_lmwt`; do
      mv $dir/scoring_$wip/$lmwt.tra $dir/scoring_$wip/$lmwt.tra.orig
      awk '{ printf("%s ",$1); for(i=NF; i>1; i--){ printf("%s ",$i); } printf("\n"); }' \
         <$dir/scoring_$wip/$lmwt.tra.orig >$dir/scoring_$wip/$lmwt.tra
    done
  fi


  [ -x local/wer_hyp_filter ] && filtering_cmd="local/wer_hyp_filter"
  for lmwt in `seq $min_lmwt $max_lmwt`; do
    utils/int2sym.pl -f 2- $symtab <$dir/scoring_$wip/$lmwt.tra | \
      $filtering_cmd  > $dir/scoring_$wip/$lmwt.txt || exit 1;
  done

  $cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring_$wip/log/score.LMWT.log \
     cat $dir/scoring_$wip/LMWT.txt \| \
      compute-wer --text --mode=present \
       ark:$dir/scoring_$wip/test_filt.txt  ark,p:- ">&" $dir/wer_LMWT_$wip || exit 1;

  if $stats; then
    mkdir -p $dir/wer-details_$wip
    $cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring_$wip/log/stats.LMWT.log \
     cat $dir/scoring_$wip/LMWT.txt \| \
       align-text --special-symbol="'***'" ark:$dir/scoring_$wip/test_filt.txt ark:- ark,t:- \|  \
       utils/scoring/wer_per_utt_details.pl --special-symbol "'***'" \| tee $dir/wer-details_$wip/per_utt_LMWT \|\
       utils/scoring/wer_per_spk_details.pl $data/utt2spk \> $dir/wer-details_$wip/per_spk_LMWT || exit 1;


    $cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring_$wip/log/stats2.LMWT.log \
       cat $dir/wer-details_$wip/per_utt_LMWT \| utils/scoring_$wip/wer_ops_details.pl --special-symbol "'***'" \| \
        column -t \| sort \> $dir/wer-details_$wip/ops_LMWT || exit 1;
  fi
done

#If we got here, the scoring was successful.
#As a  small aid to prevent confusion, we remove all wer_{?,??} files and the scoring/ dir
#These originate from the previous version of the scoring files 
rm -f $dir/wer_{?,??}
rm -rf $dir/scoring

exit 0;
