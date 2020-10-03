#!/bin/bash
# Apache 2.0

# This script computes the best path with various penalty values and LMWT.
# It will not compute the WER nor CER since the text is not  provided.


[ -f ./path.sh ] && . ./path.sh

# begin configuration section.
cmd=run.pl
beam=6
stage=0
wip=0.0
min_lmwt=7
max_lmwt=17
#end configuration section.

echo "$0 $*"  # Print the command line for logging
[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -ne 4 ]; then
  echo "Usage: $0 [--cmd (run.pl|queue.pl...)] <data-dir> <lang-dir|graph-dir> <decode-dir> <output-dir>"
  echo " Options:"
  echo "    --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes."
  echo "    --min_lmwt <int>                # minumum LM-weight for lattice rescoring "
  echo "    --max_lmwt <int>                # maximum LM-weight for lattice rescoring "
  exit 1;
fi

data=$1
lang_or_graph=$2
decode_dir=$3
output_dir=$4

symtab=$lang_or_graph/words.txt

for f in $symtab $decode_dir/lat.1.gz; do
  [ ! -f $f ]  && echo "$0: no such file $f" && exit 1;
done

hyp_filtering_cmd="cat"
[ -x local/wer_output_filter ] && hyp_filtering_cmd="local/wer_output_filter"
[ -x local/wer_hyp_filter ] && hyp_filtering_cmd="local/wer_hyp_filter"

mkdir -p $output_dir/scoring_kaldi/penalty_$wip/log
$cmd LMWT=$min_lmwt:$max_lmwt $output_dir/scoring_kaldi/penalty_$wip/log/best_path.LMWT.log \
  lattice-scale --inv-acoustic-scale=LMWT "ark:gunzip -c $decode_dir/lat.*.gz|" ark:- \| \
  lattice-add-penalty --word-ins-penalty=$wip ark:- ark:- \| \
  lattice-best-path --word-symbol-table=$symtab ark:- ark,t:- \| \
  utils/int2sym.pl -f 2- $symtab \| \
	$hyp_filtering_cmd \| \
	sed 's/[[:blank:]]*$//g' '>' $output_dir/scoring_kaldi/penalty_${wip}/LMWT.txt || exit 1;

exit 0;
