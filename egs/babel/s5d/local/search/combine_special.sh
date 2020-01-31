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
template=$1; shift;
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
declare -A duced

mkdir -p $output
mkdir -p $output/log

if [ -f $template/details/params.sh ] ; then
  . $template/details/params.sh
else
  echo >&2 "$0: Optimization output in $template/details/params.sh not found";
  exit 1;
fi


echo "$0: Combination config (id, weight, results) -- initial"

i=1
for elem in ${decode_dirs[@]} ; do
  if [ -f $elem ] ; then
    files[W$i]=$f
  elif [ -d $elem ]  && [ -d $elem/details ]  ; then
    files[W$i]=$elem/details/results
  elif [ -d $elem ] ; then
    tmpl=`cat $template/results_W${i}`
    echo $tmpl
    #exp/nnet3/lstm_bidirectional_sp/decode_dev10h.pem/kwset_kwlist4_10/details/results
    if [[ "$tmpl" == */details/results ]] ; then
      base=`echo $tmpl | sed 's:/details/results::g'`
      base=`basename $base`
      lmwt=${base##*_}
      tmpl_kwset=${base%_*}
      tmpl_kwset=${tmpl_kwset##*_}
    else
      echo >&2 "The template results file does not follow the naming pattern"
      exit 1
    fi
    f=${elem}_${lmwt}/details/results
    if [ ! -f $f ]; then
      echo >&2 "The file $f does not exist (check template or $template/results_W${i})"
      exit 1
    fi
    kwset=${elem##*_}
    if [ "$kwset" != "$tmpl_kwset" ] ; then
      echo >&2 "WARNING: The the kwset and the tmpl kwset do not match! ($kwset vs $tmpl_kwset) "
    fi

    files[W$i]=$f
  else
    echo >&2 "$0: The parameter\"$elem\" is not file nor directory"
  fi
  echo "  $i W$i=${params[W$i]} ${files[W$i]}"

  i=$(($i+1))

done



trials=$(cat $data/trials)


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


echo "DATA: $data"
if ! $skip_scoring && [ -f $data/hitlist ]  ; then
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

    cat ${output}/details/results | \
      utils/int2sym.pl -f 2 $data/utt.map | \
      local/search/utt_to_files.pl --flen "$flen" $data/../segments |\
      local/search/write_kwslist.pl --flen "$flen" --language "$language" \
      --kwlist-id "$kwlist_name" > ${output}/f4de/kwslist.xml

    if [ -f $rttm ] ; then
      cat $kwlist | local/search/annotate_kwlist.pl $data/categories > ${output}/f4de/kwlist.xml
      kwlist=${output}/f4de/kwlist.xml

      KWSEval -e $ecf -r $rttm -t $kwlist -a  \
          --zGlobalMeasures Optimum --zGlobalMeasures Supremum \
          -O -B -q 'Characters:regex=.*' -q 'NGramOrder:regex=.*' \
          -O -B -q 'OOV:regex=.*' -q 'BaseOOV:regex=.*' \
          -s ${output}/f4de/kwslist.xml -c -o -b -d -f  ${output}/f4de/

      local/kws_oracle_threshold.pl --duration $trials \
        ${output}/f4de/alignment.csv > ${output}/f4de/metrics.txt
    fi
  fi
fi

echo "$0: All OK"
