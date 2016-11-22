#! /bin/bash

# Copyright 2016 Vimal Manohar
# Apache 2.0

set -e
set -o pipefail
set -u

. path.sh
. cmd.sh

mic=sdm
task=sad

. parse_options.sh

RT04_DEV_ROOT=/export/corpora5/LDC/LDC2007S11
RT04_EVAL_ROOT=/export/corpora5/LDC/LDC2007S12/package/rt04_eval
RT05_EVAL_ROOT=/export/corpora5/LDC/LDC2011S06

if [ ! -f data/rt04_dev/.done ]; then
  local/make_rt_2004_dev.pl $RT04_DEV_ROOT data
  touch data/rt04_dev/.done
fi

if [ ! -f data/rt04_eval/.done ]; then
  local/make_rt_2004_eval.pl $RT04_EVAL_ROOT data
  touch data/rt04_eval/.done
fi

if [ ! -f data/rt05_eval/.done ]; then
  local/make_rt_2005_eval.pl $RT05_EVAL_ROOT data
  touch data/rt05_eval/.done
fi

mkdir -p data/local

dir=data/local/rt05_eval/$mic/$task
mkdir -p $dir

if [ $task == "stt" ]; then
  cp $RT05_EVAL_ROOT/data/reference/concatenated/rt05s.confmtg.050614.${task}.${mic}.stm $dir/stm
else
  cp $RT05_EVAL_ROOT/data/reference/concatenated/rt05s.confmtg.050614.${task}.${mic}.rttm $dir/rttm
fi

cp $RT05_EVAL_ROOT/data/indicies/expt_05s_${task}ul_eval05s_eng_confmtg_${mic}_1.uem $dir/uem
cat $dir/uem | awk '!/;;/{if (NF > 0) print $1}' | perl -pe 's/(.*)\.sph/$1/g' | sort -u > $dir/list
utils/subset_data_dir.sh --utt-list $dir/list data/rt05_eval data/rt05_eval_${mic}_${task}
[ -f $dir/stm ] && cp $dir/stm data/rt05_eval_${mic}_${task}
[ -f $dir/uem ] && cp $dir/uem data/rt05_eval_${mic}_${task}
[ -f $dir/rttm ] && cp $dir/rttm data/rt05_eval_${mic}_${task}

dir=data/local/rt04_dev/$mic/$task
mkdir -p $dir

if [ $task == "stt" ]; then
  cp $RT04_DEV_ROOT/data/reference/dev04s/concatenated/dev04s.040809.${mic}.stm $dir/stm
elif [ $task == "spkr" ]; then
  cp $RT04_DEV_ROOT/data/reference/dev04s/concatenated/dev04s.040809.${mic}.rttm $dir/rttm
else
  cat $RT04_DEV_ROOT/data/reference/dev04s/concatenated/dev04s.040809.${mic}.rttm | spkr2sad.pl | rttmSmooth.pl -s 0 > $dir/rttm
fi
cp $RT04_DEV_ROOT/data/indices/dev04s/dev04s.${mic}.uem $dir/uem
cat $dir/uem | awk '!/;;/{if (NF > 0) print $1}' | perl -pe 's/(.*)\.sph/$1/g' | sort -u > $dir/list
utils/subset_data_dir.sh --utt-list $dir/list data/rt04_dev data/rt04_dev_${mic}_${task}
[ -f $dir/stm ] && cp $dir/stm data/rt04_dev_${mic}_${task}
[ -f $dir/uem ] && cp $dir/uem data/rt04_dev_${mic}_${task}
[ -f $dir/rttm ] && cp $dir/rttm data/rt04_dev_${mic}_${task}

dir=data/local/rt04_eval/$mic/$task
mkdir -p $dir

if [ $task == "stt" ]; then
  cp $RT04_EVAL_ROOT/data/reference/eval04s/concatenated/eval04s.040511.${mic}.stm $dir/stm
elif [ $task == "spkr" ]; then
  cp $RT04_EVAL_ROOT/data/reference/eval04s/concatenated/eval04s.040511.${mic}.rttm $dir/rttm
else
  cat $RT04_EVAL_ROOT/data/reference/eval04s/concatenated/eval04s.040511.${mic}.rttm | spkr2sad.pl | rttmSmooth.pl -s 0 > $dir/rttm
fi
cp $RT04_EVAL_ROOT/data/indices/eval04s/eval04s.${mic}.uem $dir/uem
cat $dir/uem | awk '!/;;/{if (NF > 0) print $1}' | perl -pe 's/(.*)\.sph/$1/g' | sort -u > $dir/list
utils/subset_data_dir.sh --utt-list $dir/list data/rt04_eval data/rt04_eval_${mic}_${task}
[ -f $dir/stm ] && cp $dir/stm data/rt04_eval_${mic}_${task}
[ -f $dir/uem ] && cp $dir/uem data/rt04_eval_${mic}_${task}
[ -f $dir/rttm ] && cp $dir/rttm data/rt04_eval_${mic}_${task}
