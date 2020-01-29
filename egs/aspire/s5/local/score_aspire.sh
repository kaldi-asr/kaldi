#!/usr/bin/env bash

# Copyright Johns Hopkins University (Author: Daniel Povey, Vijayaditya Peddinti) 2016.  Apache 2.0.
# This script generates the ctm files, filters and scores them if an stm file is available

set -e
set -x

iter=final
min_lmwt=1
max_lmwt=20
default_lmwt=12 # see tune_hyper description for more info
word_ins_penalties=0.0,0.25,0.5,0.75,1.0
default_wip=0.0
ctm_beam=6
decode_mbr=true
cmd=run.pl
stage=1
resolve_overlaps=true
tune_hyper=true # if true:
                #    if the data set is "dev_aspire" we check for the
                #       best lmwt and word_insertion_penalty,
                #    else we use try to find the best values from dev_aspire decodes
                #         if not found we use the default values

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh || exit 1;

if [ $# -ne 5 ]; then
  echo "Usage: $0 [options] <lang-dir> <decode-dir> <actual-data-set> <segmented-data-set> <output-ctm-file>"
  echo " Options:"
  echo "    --stage (1|2|3)  # start scoring script from part-way through."
  echo "e.g.:"
  echo "$0 data/train data/lang exp/nnet3/tdnn"
  exit 1;
fi

lang=$1
decode_dir=$2
act_data_set=$3
segmented_data_set=$4
out_file=$5

model=$decode_dir/../$iter.mdl # assume model one level up from decoding dir.

mkdir -p $decode_dir/scoring
# create a python script to filter the ctm, for labels which are mapped
# to null strings in the glm or which are not accepted by the scoring server
python -c "
import sys, re
lines = map(lambda x: x.strip(), open('data/${act_data_set}/glm').readlines())
patterns = []
for line in lines:
  if re.search('=>', line) is not None:
    parts = re.split('=>', line.split('/')[0])
    if parts[1].strip() == '':
      patterns.append(parts[0].strip())
print '|'.join(patterns)
" > $decode_dir/scoring/glm_ignore_patterns

ignore_patterns=$(cat $decode_dir/scoring/glm_ignore_patterns)
echo "$0: Ignoring these patterns from the ctm ", $ignore_patterns
cat << EOF > $decode_dir/scoring/filter_ctm.py
import sys
file = open(sys.argv[1])
out_file = open(sys.argv[2], 'w')
ignore_set = "$ignore_patterns".split("|")
ignore_set.append("[noise]")
ignore_set.append("[laughter]")
ignore_set.append("[vocalized-noise]")
ignore_set.append("!SIL")
ignore_set.append("<unk>")
ignore_set.append("%hesitation")
ignore_set = set(ignore_set)
print ignore_set
for line in file:
  if line.split()[4] not in ignore_set:
    out_file.write(line)
out_file.close()
EOF

filter_ctm_command="python $decode_dir/scoring/filter_ctm.py "

if  $tune_hyper ; then
  # find the best lmwt and word_insertion_penalty based on the transcripts
  # provided for dev_aspire, for other data sets just copy the values from dev_aspire decode directories
  # or use the default values

  if [ $stage -le 1 ]; then
    if [[ "$act_data_set" =~ "dev_aspire" ]]; then
      wip_string=$(echo $word_ins_penalties | sed 's/,/ /g')
      temp_wips=($wip_string)
      $cmd WIP=1:${#temp_wips[@]} $decode_dir/scoring/log/score.wip.WIP.log \
        wips=\(0 $wip_string\) \&\& \
        wip=\${wips[WIP]} \&\& \
        echo \$wip \&\& \
        $cmd LMWT=$min_lmwt:$max_lmwt $decode_dir/scoring/log/score.LMWT.\$wip.log \
          local/multi_condition/get_ctm.sh --filter-ctm-command "$filter_ctm_command" \
            --beam $ctm_beam --decode-mbr $decode_mbr \
            --resolve-overlaps $resolve_overlaps \
            --glm data/${act_data_set}/glm --stm data/${act_data_set}/stm \
          LMWT \$wip $lang data/${segmented_data_set}_hires $model $decode_dir || exit 1;

      eval "grep Sum $decode_dir/score_{${min_lmwt}..${max_lmwt}}/penalty_{$word_ins_penalties}/*.sys"|utils/best_wer.sh 2>/dev/null
      eval "grep Sum $decode_dir/score_{${min_lmwt}..${max_lmwt}}/penalty_{$word_ins_penalties}/*.sys" | \
       utils/best_wer.sh 2>/dev/null | python -c "import sys, re
line = sys.stdin.readline()
file_name=line.split()[-1]
parts=file_name.split('/')
penalty = re.sub('penalty_','',parts[-2])
lmwt = re.sub('score_','', parts[-3])
lmfile=open('$decode_dir/scoring/bestLMWT','w')
lmfile.write(str(lmwt))
lmfile.close()
wipfile=open('$decode_dir/scoring/bestWIP','w')
wipfile.write(str(penalty))
wipfile.close()
" || exit 1;
        LMWT=$(cat $decode_dir/scoring/bestLMWT)
        word_ins_penalty=$(cat $decode_dir/scoring/bestWIP)
    fi
  fi


  if [[ "$act_data_set" =~ "test_aspire" ]] || [[ "$act_data_set" =~ "eval_aspire" ]]; then
    # check for the best values from dev_aspire decodes
    dev_decode_dir=$(echo $decode_dir|sed "s/test_aspire/dev_aspire_whole/g; s/eval_aspire/dev_aspire_whole/g")
    if [ -f $dev_decode_dir/scoring/bestLMWT ]; then
      LMWT=$(cat $dev_decode_dir/scoring/bestLMWT)
      echo "Using the bestLMWT $LMWT value found in  $dev_decode_dir"
    else
      LMWT=$default_lmwt # default LMWT in case hyper-parameter tuning results are not available
      echo "Unable to find the bestLMWT in the  dev decode dir $dev_decode_dir"
      echo "Keeping the default value $LMWT"
    fi
    if [ -f $dev_decode_dir/scoring/bestWIP ]; then
      word_ins_penalty=$(cat $dev_decode_dir/scoring/bestWIP)
      echo "Using the bestWIP $word_ins_penalty value found in  $dev_decode_dir"
    else
      word_ins_penalty=$default_wip # default WIP in case hyper-parameter tuning results are not available
      echo "Unable to find the bestWIP in the  dev decode dir $dev_decode_dir"
      echo "Keeping the default/user-specified value $word_ins_penalty"
    fi
  else
    echo "Using the default values for LMWT and word_ins_penalty"
  fi

fi

# lattice to ctm conversion and scoring.
if [ $stage -le 2 ]; then
  echo "Generating CTMs with LMWT $LMWT and word insertion penalty of $word_ins_penalty"
  local/multi_condition/get_ctm.sh --filter-ctm-command "$filter_ctm_command" \
    --beam $ctm_beam --decode-mbr $decode_mbr \
    $LMWT $word_ins_penalty $lang data/${segmented_data_set}_hires $model $decode_dir 2>$decode_dir/scoring/finalctm.LMWT$LMWT.WIP$word_ins_penalty.log || exit 1;
fi


# copy the ctms to the specified output files
if [ $stage -le 3 ]; then
  cat $decode_dir/score_$LMWT/penalty_$word_ins_penalty/ctm.filt | \
    awk '{split($1, parts, "-"); printf("%s 1 %s %s %s\n", parts[1], $3, $4, $5)}' > $out_file

  cat data/${segmented_data_set}_hires/wav.scp | \
    awk '{split($1, parts, "-"); printf("%s\n", parts[1])}' > $decode_dir/score_$LMWT/penalty_$word_ins_penalty/recording_names

  local/multi_condition/fill_missing_recordings.py \
    $out_file $out_file.submission \
    $decode_dir/score_$LMWT/penalty_$word_ins_penalty/recording_names

  echo "Generated the ctm @ $out_file.submission from the ctm file $decode_dir/score_${LMWT}/penalty_$word_ins_penalty/ctm.filt"
fi
