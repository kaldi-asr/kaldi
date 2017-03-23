#!/bin/bash
# Copyright (c) 2015, Johns Hopkins University ( Yenda Trmal <jtrmal@gmail.com> )
# License: Apache 2.0

# Begin configuration section.
language="201-haitian"
corpus=/export/babel/data/
indus=/export/babel/data/scoring/IndusDB
# End configuration section
. ./utils/parse_options.sh

set -e -o pipefail
set -o nounset                              # Treat unset variables as an error

corpus=$corpus/$language
lists=./conf/lists/$language/

corpusdir=$(find $corpus -maxdepth 1 -name "*-build" -type d) || exit 1
[ -z "$corpusdir" ] && "Corpus directory for $language not found!" && exit 1

train_dir=$(find $corpusdir -ipath "*/conversational/*" -name "training" -type d) || exit 1
[ -z "$train_dir" ] && "Corpus directory $corpusdir/*/training/ not found!" && exit 1

train_rom_dir=$(find $train_dir -name  "transcript_roman" -type d) || exit 1
echo "# include common settings for fullLP systems."
echo ". conf/common.limitedLP || exit 1;"
echo -e "\n"

echo "#speech corpora files location"
echo "train_data_dir=$train_dir"
echo "train_data_list=$lists/sub-train.list"
echo "train_nj=32"
echo -e "\n"


indusid=$(find $corpus -name "IARPA*-build" -type d)
indusid=$(basename ${indusid})
indusid=${indusid%%-build}
dataset=dev10h
dev10h_dir=$(find $corpusdir -ipath "*/conversational/*" -name "dev" -type d) || exit 1
indusdev10=$(find $indus/ -maxdepth 1 -name "$indusid*dev" -type d)
if [ -z "$indusdev10" ] ; then
  echo >&2  "IndusDB entry \"$indusid*dev\" not found -- removing the version and retrying"
  indusid=${indusid%%-v*}
  indusdev10=$(find $indus/ -maxdepth 1 -name "$indusid*dev" -type d)
fi

if [ -z "$indusdev10" ] ; then
  echo ""
else
  dev10h_rttm=$(find $indusdev10/ -name "*mitllfa3.rttm" )
  dev10h_ecf=$(find $indusdev10/ -name "*ecf.xml" )
  dev10h_stm=$(find $indusdev10/ -name "*stm" -not -name "*cond-speaker*" )
  kwlists1=$(find $indusdev10/ -name "*.kwlist.xml" | sort -V )
  kwlists2=$(find $indusdev10/ -name "*.kwlist?*.xml" | sort -V )
  kwlists="$kwlists1 $kwlists2"
fi

echo "#Radical reduced DEV corpora files location"
echo "dev2h_data_dir=$dev10h_dir"
echo "dev2h_data_list=$lists/dev.2h.list"
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
eval_dir=$(find $corpus -ipath "*-eval/*/conversational/*" -name "$dataset" -type d) || exit 1
if [ ! -z "$eval_dir" ] ; then
  indus_set=$(find $indus/ -maxdepth 1 -name "$indusid*$dataset" -type d)
  if [ -z "$indus_set" ] ; then
    eval_ecf=$(find $indus/ -maxdepth 1 -type f  -name "*$indusid*${dataset}.ecf.xml" )
    eval_kwlists1=$(find $indus -name "*$indusid*${dataset}.kwlist*.xml" | sort -V)
    eval_kwlists2=$(find $indus -name "*$indusid*${dataset}.kwlist?*.xml" | sort -V)
    eval_kwlists="$kwlists1 $kwlists2"
  else
    eval_rttm=$(find $indus_set/ -name "*mitllfa3.rttm" )
    eval_ecf=$(find $indus_set/ -name "*ecf.xml" )
    eval_stm=$(find $indus_set/ -name "*stm" -not -name "*cond-speaker*" )
    eval_kwlists1=$(find $indus -name "*.kwlist.xml" | sort -V)
    eval_kwlists2=$(find $indus -name "*.kwlist?*.xml" | sort -V)
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
  indus_set=$(find $indus/ -maxdepth 1 -name "$indusid*$dataset" -type d)
  if [ -z "$indus_set" ] ; then
    echo ""
  else
    evalpart1_rttm=$(find $indus_set/ -name "*mitllfa3.rttm" )
    evalpart1_ecf=$(find $indus_set/ -name "*ecf.xml" )
    evalpart1_stm=$(find $indus_set/ -name "*stm" -not -name "*cond-speaker*" )
    kwlists1=$(find $indus_set/ -name "*.kwlist.xml" | sort -V)
    kwlists2=$(find $indus_set/ -name "*.kwlist?*.xml" | sort -V)
    kwlists="$kwlists1 $kwlists2"
  fi
  echo "#Official post-EVAL period data files"
  echo "${dataset}_data_dir=$eval_dir"
  echo "${dataset}_data_list=$lists/${dataset}.list"
  echo "${dataset}_rttm_file=$evalpart1_rttm"
  echo "${dataset}_ecf_file=$evalpart1_ecf"
  echo "${dataset}_stm_file=$evalpart1_stm"
  echo "${dataset}_kwlists=("
  for list in $kwlists; do
    id=$(echo $list | sed 's/.*\(kwlist[0-9]*\)\.xml/\1/');
    echo "    [$id]=$list"
  done
  echo ")  # ${dataset}_kwlists"
  echo "${dataset}_nj=32"
  echo -e "\n"


  dataset=shadow
  echo "#Shadow data files"
  echo "shadow_data_dir=("
  echo "    $dev10h_dir"
  echo "    $eval_dir"
  echo ") # shadow_data_dir"
  echo "shadow_data_list=("
  echo "    $lists/dev.list"
  echo "    $lists/eval.lists"
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
unsup_dir=$(find $corpusdir -ipath "*/conversational/*" -name "$dataset" -type d) || exit 1
unsup_lists=( $lists/untranscribed-training.list $lists/sub-train.untranscribed.list)
unsup_dirs=( $unsup_dir $train_dir )
echo "#Unsupervised dataset for LimitedLP condition"
echo "unsup_data_list=("
for list in ${unsup_lists[*]}; do
  [ ! -f $list ] && echo "Unsupervised training set not found $list"
  echo "    $list";
done
echo ")  # unsup_data_list"

echo "unsup_data_dir=("
for dir in ${unsup_dirs[*]}; do
  [ ! -d $dir ] && echo "Unsupervised training data dir not found $dir"
  echo "    $dir";
done
echo ")  # unsup_data_dir"

echo "unsup_nj=32"
echo -e "\n"

lexicon=$(find $corpusdir -ipath "*/conversational/*" -name "lexicon.sub-train.txt" -type f) || exit 1
echo "lexicon_file=$lexicon"

if [ ! -z "$train_rom_dir" ] ; then
  echo "lexiconFlags=\"--romanized --oov <unk>\""
fi
echo -e "\n\n"


