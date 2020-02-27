#!/usr/bin/env bash
# Copyright 2013-2014  Johns Hopkins University (authors: Jan Trmal, Guoguo Chen, Dan Povey)
# Copyright (c) 2016, Johns Hopkins University (Yenda Trmal <jtrmal@gmail.com> )
# License: Apache 2.0

# begin configuration section.
cmd=run.pl
stage=0
nbest_final=900
nbest_small=20
extraid=
skip_scoring=false
optimize=true
duptime=52
power=1.1
ntrue_scale=
#end of configuration section

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

set -e -o pipefail
set -o nounset                              # Treat unset variables as an error

help_message="Usage: $0 [options] <data-dir> <lang-dir|graph-dir> <decode-dir1> <decode-dir2> [<decode-dir3> ... ] <out-dir>
E.g.: $0 data/dev10h.pem data/lang exp/tri6_nnet/decode_dev10h.pem/kws_10/  exp/tri6_nnet/decode_dev10h.pem/oov_kws_10/    exp/combine/dev10hx.pem
"
if [ $# -lt 5 ]; then
  printf "$help_message\n";
  exit 1;
fi


data=$1; shift;
lang=$1; shift;
output=${@: -1}  # last argument to the script
decode_dirs=( $@ )  # read the remaining arguments into an array
unset decode_dirs[${#decode_dirs[@]}-1]  # 'pop' the last argument which is odir
num_sys=${#decode_dirs[@]}  # number of systems to combine

if [ -z "$extraid" ] ; then
  data="$data/kws"
  output="$output/kws"
else
  data="$data/kwset_${extraid}"
  output="$output/kwset_${extraid}"
fi

if [ -z "$ntrue_scale" ] ; then
  ntrue_scale=$num_sys
fi

declare -A params=([PWR]=$power [NTRUE]=$ntrue_scale)
declare -A files
declare -A files_reduced

mkdir -p $output
mkdir -p $output/log

echo "$0: Combination config (id, weight, results) -- initial"

i=1
nsystems=0
for elem in ${decode_dirs[@]} ; do
  params[W$i]="0.5"
  if [ -f $elem ] ; then
    f=$(echo $elem | cut -d: -f1)
    w=$(echo $elem | cut -d: -s -f2)

    [ ! -z "$w" ] && params[W$i]="$w"
    files[W$i]=$f
    files_reduced[W$i]=$output/results.reduced.$i

  elif [ -d $elem ]  && [ -d $elem/details ]  ; then
    mtwv=$(cat $elem/details/score.txt | grep "MTWV *=" |cut -f 2 -d '=' | sed 's/ //g')
    params[W$i]="$mtwv"
    files[W$i]=$elem/details/results
    files_reduced[W$i]=$output/results.reduced.$i
  elif [ -d $elem ] ; then
    best_dir=$(find ${elem}_* -name "score.txt" \
                              -path "*$extraid*" \
                              -path "*/details/*" |\
               xargs grep "MTWV *=" | \
               sort -k2,2g -t '=' |
               tail -n 1 | \
               cut -f 1 -d ':' | \
               xargs dirname \
              )
    mtwv=$(cat $best_dir/score.txt | grep "MTWV *=" |cut -f 2 -d '=' | sed 's/ //g')
    params[W$i]="$mtwv"
    files[W$i]=$best_dir/results
    files_reduced[W$i]=$output/results.reduced.$i
  else
    echo >&2 "$0: The parameter\"$elem\" is not file nor directory"
  fi

  echo "  $i W$i=${params[W$i]} ${files[W$i]}"
  echo "${files[W$i]}" > $output/results_W$i

  cat ${files[W$i]} | \
    local/search/filter_kws_results.pl --probs --nbest $nbest_small > ${files_reduced[W$i]}

  nsystems=$i
  i=$(($i+1))

done

if [ $nsystems -le 0 ] ; then
  echo >&2 "No acoustic system found"
  return 1
fi

trials=$(cat $data/trials)

if $optimize ; then
  cmdline=


  declare -A params
  opt_vars=""
  opt_task_params=""
  for w in "${!params[@]}" ; do
    opt_vars="$opt_vars --var $w=${params[$w]}"

    if [ ${files_reduced[$w]+isset} ] ; then
      opt_task_params="$opt_task_params $w ${files_reduced[$w]}"
    fi
  done

  echo "$0: Optimization -- first stage (reduced size results)"
  mkdir -p $output/opt
  local/optimize2.pl --result-regexp '.*ATWV *= *(.*)' --ftol 0.01 --iftol 0.01\
    --output-dir $output/opt $opt_vars \
    local/search/combine_results.pl --probs --power PWR $opt_task_params - \| \
    local/search/normalize_results_kst.pl  --duration $trials --ntrue-scale NTRUE\| \
    local/search/filter_kws_results.pl --nbest 100 \| \
    compute-atwv $trials ark:$data/hitlist ark:- | \
    tee $output/log/optimize.log | grep -i "Iter" || {
      echo >&2 "$0: Optimization failed (see $output/log/optimize.log for errors)"; exit 1
    }

  # override the default parameters
  if [ -f $output/opt/params.sh ] ; then
    . $output/opt/params.sh
  else
    echo >&2 "$0: Optimization output in $output/opt/params.sh not found";
    exit 1;
  fi

  # Second round of optimization -- this time, only the NTRUE
  comb_task_params=""
  for w  in "${!params[@]}" ; do
    if [ ${files[$w]+isset} ] ; then
      comb_task_params="$comb_task_params ${params[$w]} ${files[$w]}"
    fi
  done

  echo "$0: Optimization -- second stage (full size results)"
  mkdir -p $output/opt_ntrue
  local/optimize2.pl --result-regexp '.*ATWV *= *(.*)' \
    --output-dir $output/opt_ntrue --var NTRUE=${params[NTRUE]}  \
    local/search/combine_results.pl --probs --tolerance $duptime --power ${params[PWR]}  $comb_task_params - \| \
    local/search/normalize_results_kst.pl  --duration $trials --ntrue-scale NTRUE\| \
    local/search/filter_kws_results.pl --probs --duptime $duptime  \| \
    compute-atwv $trials ark:$data/hitlist ark:- | \
    tee $output/log/optimize_ntrue.log | grep -i "Iteration" || {
      echo >&2 "$0: Optimization failed (see $output/log/optimize_ntrue.log for errors)"; exit 1
    }
  # override the default parameters
  if [ -f $output/opt_ntrue/params.sh ] ; then
    . $output/opt_ntrue/params.sh
  else
    echo >&2 "$0: Optimization output in $output/opt_ntrue/params.sh not found";
    exit 1;
  fi
fi

echo "$0: Combination config (final)"
echo -n "$0:   params=["
comb_task_params=""
for w  in "${!params[@]}" ; do
  echo -n  " $w=${params[$w]}"
  if [ ${files[$w]+isset} ] ; then
    comb_task_params="$comb_task_params ${params[$w]} ${files[$w]}"
  fi
done
echo "]"

mkdir -p $output/details


echo "$0: Doing final combination"
local/search/combine_results.pl \
    --probs --tolerance $duptime --power ${params[PWR]} $comb_task_params - | \
  local/search/normalize_results_kst.pl  \
    --duration $trials --ntrue-scale ${params[NTRUE]} |\
  local/search/filter_kws_results.pl --probs --duptime $duptime > $output/details/results

#Write the parapeters
echo "declare -A params" > $output/details/params.sh
for w  in "${!params[@]}" ; do
  echo "params[$w]=${params[$w]}"
done >> $output/details/params.sh
echo "${params[NTRUE]}" > $output/details/ntrue
echo "${params[PWR]}" > $output/details/power

if ! $skip_scoring ; then
  echo "$0: Scoring..."
  cat $output/details/results |\
    compute-atwv $trials ark,t:$data/hitlist ark:- \
      ${output}/details/alignment.csv \
       > ${output}/details/score.txt \
       2> ${output}/log/score.log

  cat ${output}/details/alignment.csv |\
    perl local/search/per_category_stats.pl \
      --sweep-step 0.005  $trials $data/categories \
      > ${output}/details/per-category-score.txt \
      2> ${output}/log/per-category-score.log

  cp $output/details/score.txt $output/score.txt

fi

if [ $stage -le 2 ]; then
  if [ -f $data/f4de_attribs ] ; then
    language=""
    flen=0.01
    kwlist_name=""
    . $data/f4de_attribs #override the previous variables

    ecf=$data/ecf.xml
    rttm=$data/rttm
    kwlist=$data/kwlist.xml

    mkdir -p ${output}/f4de/

    cat $kwlist | local/search/annotate_kwlist.pl $data/categories > ${output}/f4de/kwlist.xml
    kwlist=${output}/f4de/kwlist.xml

    cat ${output}/details/results | \
      utils/int2sym.pl -f 2 $data/utt.map | \
      local/search/utt_to_files.pl --flen "$flen" $data/../segments |\
      local/search/write_kwslist.pl --flen "$flen" --language "$language" \
      --kwlist-id "$kwlist_name" > ${output}/f4de/kwslist.xml

    KWSEval -e $ecf -r $rttm -t $kwlist -a  \
        --zGlobalMeasures Optimum --zGlobalMeasures Supremum \
        -O -B -q 'Characters:regex=.*' -q 'NGramOrder:regex=.*' \
        -O -B -q 'OOV:regex=.*' -q 'BaseOOV:regex=.*' \
        -s ${output}/f4de/kwslist.xml -c -o -b -d -f  ${output}/f4de/

    local/kws_oracle_threshold.pl --duration $trials \
      ${output}/f4de/alignment.csv > ${output}/f4de/metrics.txt
  fi
fi

echo "$0: All OK"
