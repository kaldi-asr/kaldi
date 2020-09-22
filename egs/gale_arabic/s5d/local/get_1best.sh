#!/usr/bin/env bash

# begin configuration section.
cmd=run.pl
wip=0.0
min_lmwt=7
max_lmwt=17
weight=0.5
# end configuration section.

echo "$0 $*" # Printing the command line for logging
[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -ne 5 ]; then
  echo "Usage: $0 [--cmd (run.pl|queue.pl...)] <lang-dir> <decode-dir> <dir> <output-dir>"
  exit 1;
fi

data_dir=$1
lang_dir=$2
decode_dir=$3
dir=$4
output_file=$5


hyp_filtering_cmd="cat"
symtab=${lang_dir}/words.txt

$cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring_kaldi/penalty_${wip}/log/best_path.LMWT.log \
  lattice-scale --inv-acoustic-scale=LMWT "ark:gunzip -c ${decode_dir}_rnnlm_${weight}/lat.*.gz|" ark:- \| \
  lattice-add-penalty --word-ins-penalty=${wip} ark:- ark:- \| \
  lattice-best-path --word-symbol-table=$symtab ark:- ark,t:- \| \
  utils/int2sym.pl -f 2- $symtab \| \
  $hyp_filtering_cmd '>' ${decode_dir}_rnnlm_${weight}/scoring_kaldi/penalty_${wip}_rnnlm/LMWT.txt || exit 1;

for f in ${decode_dir}_rnnlm_${weight}/scoring_kaldi/penalty_${wip}_rnnlm/*.txt; do
  lmwt=$(basename $f | cut -d '.' -f1)
  cut -d ' ' -f1 ${f} > ${data_dir}/tmp/id
  cut -d ' ' -f2- ${f} > ${data_dir}/tmp/chars
  perl local/eng2arabic.pl ${data_dir}/tmp/chars ${data_dir}/tmp/Arabic
  paste -d ' ' ${data_dir}/tmp/id ${data_dir}/tmp/Arabic > ${output_file}
done

exit 0;
