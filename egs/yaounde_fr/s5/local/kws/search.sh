#!/bin/bash
# Copyright 2012-2018  Johns Hopkins University (Author: Guoguo Chen, Yenda Trmal)
# License: Apache 2.0


help_message="$(basename $0): do keyword indexing and search.  data-dir is assumed to have
                 kws/ subdirectory that specifies the terms to search for.  Output is in
                 decode-dir/kws/
             Usage:
                 $(basename $0) <lang-dir> <data-dir> <decode-dir>"

# Begin configuration section.
min_lmwt=8
max_lmwt=12
cmd=run.pl
model=
skip_scoring=false
skip_optimization=false # true can speed it up if #keywords is small.
max_states=350000
indices_dir=
kwsout_dir=
stage=0
word_ins_penalty=0
extraid=
silence_word=  # specify this if you did to in kws_setup.sh, it's more accurate.
strict=false
duptime=0.6
ntrue_scale=1.0
frame_subsampling_factor=1
nbest=-1
max_silence_frames=50
skip_indexing=false
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

set -u
set -e
set -o pipefail


if [[ "$#" -ne "3" ]] ; then
    echo -e "$0: FATAL: wrong number of script parameters!\n\n"
    printf "$help_message\n\n"
    exit 1;
fi

silence_opt=

langdir=$1
datadir=$2
decodedir=$3

if [ -z $extraid ] ; then
  kwsdatadir=$datadir/kws
else
  kwsdatadir=$datadir/kwset_${extraid}
fi

if [ -z $extraid ] ; then
  kwsoutdir=$decodedir/kws
else
  kwsoutdir=$decodedir/kwset_${extraid}
fi


if [ -z $indices_dir ]; then
  indices_dir=$kwsoutdir
fi

if [ ! -z "$model" ]; then
    model_flags="--model $model"
else
    model_flags=
fi

mkdir -p $kwsoutdir
for d in "$datadir" "$kwsdatadir" "$langdir" "$decodedir"; do
  if [ ! -d "$d" ]; then
    echo "$0: FATAL: expected directory $d to exist"
    exit 1;
  fi
done

echo "$0: Searching: $kwsdatadir"
duration=$(cat $kwsdatadir/trials)
echo "$0: Duration: $duration"


frame_subsampling_factor=1
if [ -f $decodedir/../frame_subsampling_factor ] ; then
  frame_subsampling_factor=$(cat $decodedir/../frame_subsampling_factor)
  echo "$0: Frame subsampling factor autodetected: $frame_subsampling_factor"
elif [ -f $decodedir/../../frame_subsampling_factor ] ; then
  frame_subsampling_factor=$(cat $decodedir/../../frame_subsampling_factor)
  echo "$0: Frame subsampling factor autodetected: $frame_subsampling_factor"
fi

if [ $stage -le 0 ] ; then
  if [ ! -f $indices_dir/.done.index ] && ! $skip_indexing ; then
    [ ! -d $indices_dir ] && mkdir  $indices_dir
    for lmwt in $(seq $min_lmwt $max_lmwt) ; do
      indices=${indices_dir}_$lmwt
      mkdir -p $indices

      acwt=$(perl -e "print 1.0/$lmwt")
      [ ! -z $silence_word ] && silence_opt="--silence-word $silence_word"
      steps/make_index.sh $silence_opt --cmd "$cmd" --acwt $acwt $model_flags\
        --skip-optimization $skip_optimization --max-states $max_states \
        --word-ins-penalty $word_ins_penalty --max-silence-frames $max_silence_frames\
        --frame-subsampling-factor ${frame_subsampling_factor} \
        $kwsdatadir $langdir $decodedir $indices  || exit 1
    done
    touch $indices_dir/.done.index
  else
    echo "$0: Assuming indexing has been aready done. If you really need to re-run "
    echo "$0: the indexing again, delete the file $indices_dir/.done.index"
  fi
fi

keywords=$kwsdatadir/keywords.fsts
if [ -f $keywords ] ; then
  echo "$0: Using ${keywords} for search"
  keywords="ark:$keywords"
elif [ -f ${keywords}.gz ] ; then
  echo "$0: Using ${keywords}.gz for search"
  keywords="ark:gunzip -c ${keywords}.gz |"
else
  echo "$0: The keyword file ${keywords}[.gz] does not exist"
fi


if [ $stage -le 1 ]; then
  for lmwt in $(seq $min_lmwt $max_lmwt) ; do
    kwsoutput=${kwsoutdir}_$lmwt
    indices=${indices_dir}_$lmwt
    nj=$(cat $indices/num_jobs)


    for f in $indices/index.1.gz ; do
      [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
    done

    mkdir -p $kwsoutput/log
    $cmd JOB=1:$nj $kwsoutput/log/search.JOB.log \
      set -e  -o pipefail '&&' \
      kws-search --strict=$strict --negative-tolerance=-1 \
      --frame-subsampling-factor=${frame_subsampling_factor} \
      "ark:gzip -cdf $indices/index.JOB.gz|" "$keywords" \
      "ark,t:| sort -u | gzip -c > $kwsoutput/result.JOB.gz" \
      "ark,t:| sort -u | gzip -c > $kwsoutput/stats.JOB.gz" || exit 1;
  done
fi

if [ $stage -le 2 ]; then
  for lmwt in $(seq $min_lmwt $max_lmwt) ; do
    kwsoutput=${kwsoutdir}_$lmwt
    indices=${indices_dir}_$lmwt
    nj=$(cat $indices/num_jobs)

    # This is a memory-efficient way how to do the filtration
    # we do this in this way because the result.* files can be fairly big
    # and we do not want to run into troubles with memory
    files=""
    for job in $(seq 1 $nj); do
      if [ -f $kwsoutput/result.${job}.gz ] ; then
       files="$files <(gunzip -c $kwsoutput/result.${job}.gz)"
      elif [ -f $kwsoutput/result.${job} ] ; then
       files="$files $kwsoutput/result.${job}"
      else
        echo >&2 "The file $kwsoutput/result.${job}[.gz] does not exist"
        exit 1
      fi
    done
    # we have to call it using eval as we need the bash to interpret
    # the (possible) command substitution in case of gz files
    # bash -c would probably work as well, but would spawn another
    # shell instance
    eval "sort -m -u $files" |\
      local/kws/filter_kws_results.pl --likes --nbest $nbest > $kwsoutput/results || exit 1
  done
fi

if [ -z $extraid ] ; then
  extraid_flags=
else
  extraid_flags="  --extraid ""$extraid"" "
fi

if [ $stage -le 4 ]; then
  if $skip_scoring ; then
    echo "$0: Not scoring, because --skip-scoring true was issued"
  elif [ ! -x local/kws/score.sh ] ; then
    echo "$0: Not scoring, because the file local/kws_score.sh is not present"
  else
    echo "$0: Scoring KWS results"
    local/kws/score.sh --cmd "$cmd" \
      --min-lmwt $min_lmwt --max-lmwt $max_lmwt $extraid_flags \
      $langdir $datadir ${kwsoutdir} || exit 1;
  fi
fi

echo "$0: Done"
exit 0

