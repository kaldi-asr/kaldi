#!/usr/bin/env bash
# Copyright (c) 2015, Johns Hopkins University ( Yenda Trmal <jtrmal@gmail.com> )
# License: Apache 2.0

# Begin configuration section.
language="201-haitian"
corpus=/export/babel/data
indus=/export/babel/data/scoring/IndusDB
# End configuration section

echo >&2 "$0 $@"
. ./utils/parse_options.sh

cmdline="$0 --language \"$language\" --corpus \"$corpus\"  --indus \"$indus\""

set -e -o pipefail
set -o nounset                              # Treat unset variables as an error

[ -z "$language" ] && echo >&2 "Language code cannot be empty!" && exit 1
corpus=$corpus/$language
lists=./conf/lists/$language

corpusdir=$(find -L $corpus -maxdepth 1 \( -name "release-current-b" \) \( -type d -o -type l \) )
[ -z "$corpusdir" ] && corpusdir=$(find -L $corpus -maxdepth 1 \( -name "release-current"  \) \( -type d -o -type l \) )
[ -z "$corpusdir" ] && corpusdir=$(find -L $corpus -maxdepth 1 -name "*-build" -type d)
[ -z "$corpusdir" ] && echo >&2 "Corpus directory for $language not found!" && exit 1

train_dir=$(find -L $corpusdir -ipath "*/conversational/*" -name "training" -type d) || exit 1
[ -z "$train_dir" ] && echo >&2 "Corpus directory $corpusdir/*/training/ not found!" && exit 1

[ ! -f "$lists/sub-train.list" ] && [ ! -f "$lists/train.LimitedLP.list" ] &&  echo >&2 "List file not found $lists/sub-train.list nor $lists/train.LimitedLP.list" && exit 1
[ ! -f "$lists/dev.2h.list" ] && [ ! -f "$lists/dev.list" ] &&  echo >&2 "List file not found $lists/dev.2h.list nor $lists/dev.list" && exit 1
[ ! -f "$lists/dev.list" ] && echo >&2 "List file not found $lists/dev.list" && exit 1

train_rom_dir=$(find -L $train_dir -name  "transcript_roman" -type d) || exit 1
echo "# config file generated automatically by calling"
echo "#   $cmdline"
echo -e "\n"
echo "# include common settings for fullLP systems."
echo ". conf/common.fullLP || exit 1;"
echo -e "\n"

echo "#speech corpora files location"
echo "train_data_dir=$train_dir"
if [ -f "$lists/training.list" ] ; then
  echo "train_data_list=$lists/training.list"
elif [ -f "$lists/train.FullLP.list" ] ; then
  echo "train_data_list=$lists/train.FullLP.list"
else
  echo >&2 "Training list $lists/training.list not found"
fi

echo "train_nj=32"
echo -e "\n"


indusid=$(find -L $corpus -name "IARPA*-build" -type d)
[ -z "$indusid" ] && indusid=$(find -L $corpus \( -name "release-current" -o -name "release-current-b" \) -type d)
[ -z "$indusid" ] && echo >&2 "Didn't find anything that could be used as IndusDB id"  && exit 1

indusid=$(basename ${indusid})
indusid=${indusid%%-build}
dataset=dev10h
dev10h_dir=$(find -L $corpusdir -ipath "*/conversational/*" -name "dev" -type d) || exit 1
indusdev10=$(find -L $indus/ -maxdepth 1 -name "$indusid*dev" -type d)
if [ -z "$indusdev10" ] ; then
  echo >&2  "IndusDB entry \"$indusid*dev\" not found -- removing the version and retrying"
  indusid=${indusid%%-v*}
  indusdev10=$(find -L $indus/ -maxdepth 1 -name "$indusid*dev" -type d)
  if [ -z "$indusdev10" ] ; then
    echo >&2  "IndusDB entry \"$indusid*dev\" not found -- keeping only the language code and retrying"
    indusid=${language%%-*}
    indusdev10=$(find -L $indus/ -maxdepth 1 -name "*${indusid}*dev" -type d)
    if [ -z "$indusdev10" ] ; then
      echo >&2 "IndusDB configuration for the language code $indusid not found"
      exit 1
    else
      echo >&2 "IndusDB configuration found: $indusdev10 "
    fi
  fi
fi

if [ -z "$indusdev10" ] ; then
  echo ""
else
  dev10h_rttm=$(find -L $indusdev10/ -name "*mitllfa3.rttm" )
  dev10h_ecf=$(find -L $indusdev10/ -name "*ecf.xml" -not -name "*cond-*")
  dev10h_stm=$(find -L $indusdev10/ -name "*stm" -not -name "*cond-speaker*" )
  kwlists1=$(find -L $indusdev10/ -name "*.kwlist.xml" | sort -V )
  kwlists2=$(find -L $indusdev10/ -name "*.kwlist?*.xml" | sort -V )
  kwlists="$kwlists1 $kwlists2"
  dev10h_kwlists="$kwlists"
fi

echo "#Radical reduced DEV corpora files location"
echo "dev2h_data_dir=$dev10h_dir"
[  -f "$lists/dev.2h.list" ] && echo "dev2h_data_list=$lists/dev.2h.list"
[ ! -f "$lists/dev.2h.list" ] && echo >&2 "Dev2h split not found ($lists/dev.2h.list), using dev10h split ($lists/dev.list)" && echo "dev2h_data_list=$lists/dev.list"

[ ! -z ${dev10h_rttm:-}  ] && echo "dev2h_rttm_file=$dev10h_rttm"
[ ! -z ${dev10h_ecf:-} ] && echo "dev2h_ecf_file=$dev10h_ecf"
[ ! -z ${dev10h_stm:-} ] && echo "dev2h_stm_file=$dev10h_stm"
if [ ! -z "${kwlists:-}" ] ; then
  echo "dev2h_kwlists=("
  for list in $kwlists; do
    id=$(echo $list | sed 's/.*\(kwlist[0-9]*\)\.xml/\1/');
    echo "    [$id]=$list"
  done
  echo ")  # dev2h_kwlists"
fi
echo "dev2h_nj=16"
echo "dev2h_subset_ecf=true"
echo -e "\n"

echo "#Official DEV corpora files location"
echo "dev10h_data_dir=$dev10h_dir"
echo "dev10h_data_list=$lists/dev.list"
[ ! -z ${dev10h_rttm:-} ] && echo "dev10h_rttm_file=$dev10h_rttm"
[ ! -z ${dev10h_ecf:-} ]  && echo "dev10h_ecf_file=$dev10h_ecf"
[ ! -z ${dev10h_stm:-} ]  && echo "dev10h_stm_file=$dev10h_stm"
if [ ! -z "${kwlists:-}" ] ; then
  echo "dev10h_kwlists=("
  for list in $kwlists; do
    id=$(echo $list | sed 's/.*\(kwlist[0-9]*\)\.xml/\1/');
    echo "    [$id]=$list"
  done
  echo ")  # dev10h_kwlists"
fi
echo "dev10h_nj=32"
echo -e "\n"

dataset="eval"
eval_dir=$(find -L $corpus -ipath "*-eval/*/conversational/*" -name "$dataset" -type d -print -quit) || exit 1
[ -z "$eval_dir" ] && { eval_dir=$(find -L $corpusdir -ipath "*/conversational/*" -name "eval" -type d) || exit 1; }
if [ ! -z "$eval_dir" ] ; then
  indus_set=$(find -L $indus/ -maxdepth 1 -name "$indusid*$dataset" -type d)
  if [ -z "$indus_set" ] ; then
    eval_ecf=$(find -L $indus/ -maxdepth 1 -type f  -name "*$indusid*${dataset}.ecf.xml" -not -iname "*dryrun*" )
    eval_kwlists1=$(find -L $indus -name "*$indusid*${dataset}.kwlist.xml" | sort -V)
    eval_kwlists2=$(find -L $indus -name "*$indusid*${dataset}.kwlist?*.xml" | sort -V)
    eval_kwlists="$kwlists1 $kwlists2"
  else
    eval_rttm=$(find -L $indus_set/ -name "*mitllfa3.rttm" )
    eval_ecf=$(find -L $indus_set/ -name "*ecf.xml" -not -iname "*dryrun*")
    eval_stm=$(find -L $indus_set/ -name "*stm" -not -name "*cond-speaker*" )
    eval_kwlists1=$(find -L $indus -name "*.kwlist.xml" | sort -V)
    eval_kwlists2=$(find -L $indus -name "*.kwlist?*.xml" | sort -V)
    eval_kwlists="$kwlist1 $kwlist2"
  fi
  echo "#Official EVAL period evaluation data files"
  echo "eval_data_dir=$eval_dir"
  echo "eval_data_list=$lists/eval.list"
  echo "${dataset}_ecf_file=$eval_ecf"
  echo "${dataset}_kwlists=("
  for list in $eval_kwlists; do
    id=$(echo $list | sed 's/.*\(kwlist[0-9]*\)\.xml/\1/');
    echo "    [$id]=$list"
  done
  echo ")  # ${dataset}_kwlists"
  echo "eval_nj=32"
  echo -e "\n"

  dataset=evalpart1
  indus_set=$(find -L $indus/ -maxdepth 1 -name "$indusid*$dataset" -type d)
  if [ -z "$indus_set" ] ; then
    echo >&2  "IndusDB entry \"$indusid*$dataset\" not found -- keeping only the language code and retrying"
    indusid=${language%%-*}
    indus_set=$(find -L $indus/ -maxdepth 1 -name "*${indusid}*$dataset" -type d)
    if [ -z "$indus_set" ] ; then
      echo >&2 "IndusDB configuration for the language code $indusid*$dataset not found"
    else
      echo >&2 "IndusDB configuration found: $indus_set"
    fi
  fi
  if [ ! -z "$indus_set" ] ; then
    evalpart1_rttm=$(find -L $indus_set/ -name "*mitllfa3.rttm" )
    evalpart1_ecf=$(find -L $indus_set/ -name "*ecf.xml" )
    evalpart1_stm=$(find -L $indus_set/ -name "*stm" -not -name "*cond-speaker*" )
    kwlists1=$(find -L $indus_set/ -name "*.kwlist.xml" | sort -V)
    kwlists2=$(find -L $indus_set/ -name "*.kwlist?*.xml" | sort -V)
    kwlists="$kwlists1 $kwlists2"

    kwlists="$dev10h_kwlists $eval_kwlists $kwlists"
    echo "#Official post-EVAL period data files"
    echo "${dataset}_data_dir=$eval_dir"
    echo "${dataset}_data_list=$lists/${dataset}.list"
    echo "${dataset}_rttm_file=$evalpart1_rttm"
    echo "${dataset}_ecf_file=$evalpart1_ecf"
    echo "${dataset}_stm_file=$evalpart1_stm"
    echo "${dataset}_kwlists=("
    declare -A tmp_kwlists;
    for list in $kwlists; do
      id=$(echo $list | sed 's/.*\(kwlist[0-9]*\)\.xml/\1/');
      tmp_kwlists[$id]="$list"
    done

    indices=$(
      for id in "${!tmp_kwlists[@]}"; do
        echo $id
      done | sort -V | paste -s
    )
    for id in $indices; do
      echo "    [$id]=${tmp_kwlists[$id]}"
    done
    echo ")  # ${dataset}_kwlists"
    echo "${dataset}_nj=32"
    echo -e "\n"
  fi

  dataset=shadow
  echo "#Shadow data files"
  echo "shadow_data_dir=("
  echo "    $dev10h_dir"
  echo "    $eval_dir"
  echo ") # shadow_data_dir"
  echo "shadow_data_list=("
  echo "    $lists/dev.list"
  echo "    $lists/eval.list"
  echo ") # shadow_data_dir"
  echo "shadow_ecf_file=$dev10h_ecf"
  echo "shadow_rttm_file=$dev10h_rttm"
  echo "shadow_stm_file=$dev10h_stm"
  echo "shadow_kwlists=("
  for list in $eval_kwlists; do
    id=$(echo $list | sed 's/.*\(kwlist[0-9]*\)\.xml/\1/');
    echo "    [$id]=$list"
  done
  echo ")  # shadow_kwlists"
  echo "shadow_nj=32"
  echo -e "\n"
fi

dataset=untranscribed-training
unsup_dir=$(find -L $corpusdir -ipath "*/conversational/*" -name "$dataset" -type d) || exit 1
unsup_list=$lists/untranscribed-training.list
[ ! -f $unsup_list ] && echo >&2 "Unsupervised training set not found $unsup_list"
if [ -f $unsup_list ] ; then
  echo "#Unsupervised dataset for FullLP condition"
  echo "unsup_data_dir=$unsup_dir"
  echo "unsup_data_list=$unsup_list"
  echo "unsup_nj=32"
  echo -e "\n"
else
  echo "#Unsupervised training set file ($unsup_list) not found."
fi

lexicon=$(find -L $corpusdir -ipath "*/conversational/*" -name "lexicon.txt" -type f) || exit 1
echo "lexicon_file=$lexicon"

if [ ! -z "$train_rom_dir" ] ; then
  echo "lexiconFlags=\"--romanized --oov <unk>\""
fi
echo -e "\n\n"


