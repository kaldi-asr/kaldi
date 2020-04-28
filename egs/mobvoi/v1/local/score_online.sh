#!/bin/bash
# Copyright 2018-2019  Daniel Povey
#           2018-2020  Yiming Wang
# Apache 2.0

[ -f ./path.sh ] && . ./path.sh

# begin configuration section.
wake_word="嗨小问"
#end configuration section.

echo "$0 $@"  # Print the command line for logging
[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
  echo "Usage: $0 <data-dir> <lang-dir|graph-dir> <decode-dir>"
  echo " Options:"
  exit 1;
fi

data=$1
lang_or_graph=$2
dir=$3

symtab=$lang_or_graph/words.txt

for f in $symtab $data/text; do
  [ ! -f $f ] && echo "score.sh: no such file $f" && exit 1;
done


utils/data/get_utt2dur.sh $data
rm $data/utt2dur_negative 2>/dev/null || true
utils/filter_scp.pl <(grep -v $wake_word $data/text) $data/utt2dur > $data/utt2dur_negative && dur=`awk '{a+=$2} END{print a}' $data/utt2dur_negative`
echo "total duration (in seconds) of negative examples in $data: $dur"

ref_filtering_cmd="cat"
[ -x local/wer_output_filter ] && ref_filtering_cmd="local/wer_output_filter"
[ -x local/wer_ref_filter ] && ref_filtering_cmd="local/wer_ref_filter"
hyp_filtering_cmd="cat"
[ -x local/wer_output_filter ] && hyp_filtering_cmd="local/wer_output_filter"
[ -x local/wer_hyp_filter ] && hyp_filtering_cmd="local/wer_hyp_filter"


mkdir -p $dir/scoring_kaldi
cat $data/text | $ref_filtering_cmd > $dir/scoring_kaldi/test_filt.txt || exit 1;
cat $dir/trans.txt | utils/int2sym.pl -f 2- $symtab | $hyp_filtering_cmd > $dir/scoring_kaldi/hyp_filt.txt || exit 1;
export LC_ALL=en_US.UTF-8
cat $dir/scoring_kaldi/hyp_filt.txt | \
local/compute_metrics.py $dir/scoring_kaldi/test_filt.txt - --wake-word $wake_word \
  --duration $dur > $dir/scoring_kaldi/all_results
export LC_ALL=C

exit 0;
