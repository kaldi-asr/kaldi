#!/usr/bin/env bash
# Copyright 2016  Xiaohui Zhang  Apache 2.0.
# Copyright 2019  SmartAction (kkm)
# Copyright 2019  manhong wang (marvin)

# This script only combines transform file in the aligments dirs, egs: trans.1,  and
# validates matching of the utterances and alignments after combining. you would need this fmllr trans
# files after you combine ali or lat dirs(combine_ali_dirs.sh or combine_lat_dis.sh).

# Begin configuration section.
cmd=run.pl
tolerance=10
# End configuration section.
echo "$0 $@"  # Print the command line for logging.

[[ -f path.sh ]] && . ./path.sh
. parse_options.sh || exit 1

export LC_ALL=C

if [[ $# -lt 3 ]]; then
  cat >&2 <<EOF
Usage: $0 [options] <data> <dest-dir> <src-dir1> <src-dir2> ...
 e.g.: $0 data/train exp/tri3_trans_combined exp/tri3_trans_1 exp_tri3_trans_2
Options:
 --tolerance <int,%>    # maximum percentage of missing trans
                        # w.r.t. total utterances in <data> before error is
                        # reported [10]

Note:we do not checks that certain important files are present and compatible in all
source directories (phones.txt, tree) here.Because you would run combine_trans_dirs.sh 
or combine_lat_dis.sh first.

EOF
  exit 1;
fi


data=$1
dest=$2
shift 2
first_src=$1

do_trans=true    


# All checks passed, ok to prepare directory. but we do not Copy model and other files from
# the first source.

for src in $@; do
  if [[ "$(cd 2>/dev/null -P -- "$src" && pwd)" = \
        "$(cd 2>/dev/null -P -- "$dest" && pwd)" ]]; then
    echo "$0: error: Source $src is same as target $dest."
    exit 1
  fi
  if $do_trans && [[ ! -f $src/trans.1 ]]; then
    echo "$0: warning: transform (trans.*) are not present in $src, not" \
         "combining. please check you files" 
    exit 1
  fi
done

if [ ! -f $dest/ali.1.gz  ] && [ ! -f $dest/lat.1.gz ] ; then 
    echo "$0: warning: we assume you have combined the ali or lat dirs " \
         "please run combine_ali_dir.sh or combine_lat_dir.sh firstly"
    exit 1
fi

nj=$(cat $dest/num_jobs)

if [ -f $dest/trans.1 ] ; then rm $dest/trans.* ;fi    #remove old trans.*

# Make temporary directory, delete on signal, but not on 'exit 1'.
temp_dir=$(mktemp -d $dest/temp.XXXXXX) || exit 1
cleanup() { rm -rf "$temp_dir"; }
trap cleanup HUP INT TERM
echo "$0: note: Temporary directory $temp_dir will not be deleted in case of" \
     "script failure, so you could examine it for troubleshooting."

do_combine_trans() {
  local ark=$1 entities=$2 copy_program=$3
  shift 3

  echo "$0: Gathering $entities from each source directory."
  # Assign all source gzipped archive names to an exported variable, one each
  # per source directory, so that we can copy archives in a job per source.
  src_id=0
  for src in $@; do
    src_id=$((src_id + 1))
    nj_src=$(cat $src/num_jobs) || exit 1
    # Create and export variable src_arcs_${src_id} for the job runner.
    # Each numbered variable will contain the list of archives, e. g.:
    # src_arcs_1="exp/tri3_ali/trans.1 exp/tri3_ali/trans.1 ..."
    # ('printf' repeats its format as long as there are more arguments).
    printf "$src/$ark.%d " $(seq $nj_src) > $temp_dir/src_arks.${src_id}
  done
  
  # Gather archives in parallel jobs.
  $cmd JOB=1:$src_id $dest/log/gather_$entities.JOB.log \
    $copy_program \
      "ark:cat \$(cat $temp_dir/src_arks.JOB) |" \
      "ark,scp:$temp_dir/$ark.JOB,$temp_dir/$ark.JOB.scp" || exit 1

  # Merge (presumed already sorted) scp's into a single script.
  sort -m $temp_dir/$ark.*.scp > $temp_dir/$ark.scp || exit 1

  echo "$0: Splitting combined $entities into $nj archives on speaker boundary."
  $cmd JOB=1:$nj $dest/log/chop_combined_$entities.JOB.log \
    $copy_program \
      "scp:utils/split_scp.pl  -j $nj JOB --one-based $temp_dir/$ark.scp |" \
      "ark:$dest/$ark.JOB" || exit 1

  # Get some interesting stats.
  n_utt=$(wc -l <$data/spk2utt)
  n_trans=$(wc -l <$temp_dir/$ark.scp)
  n_utt_no_trans_pct=$(perl -e "print int(($n_utt - $n_trans)/$n_utt * 100 + .5);")
  echo "$0: Combined $n_trans $entities for $n_utt utterances." 

  if (( $n_utt_no_trans_pct >= $tolerance )); then
    echo "$0: error: Percentage of utterances missing $entities," \
         "${n_utt_no_trans_pct}%, is at or above error tolerance ${tolerance}%."
    exit 1
  fi

  return 0
}

$do_trans && do_combine_trans trans 'transforms' copy-matrix "$@"

cleanup     # Delete the temporary directory on success.

echo "$0: Stored combined fmllr trans in $dest"  
exit 0
