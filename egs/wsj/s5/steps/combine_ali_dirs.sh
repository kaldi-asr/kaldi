#!/usr/bin/env bash
# Copyright 2016  Xiaohui Zhang  Apache 2.0.
# Copyright 2019  SmartAction (kkm)

# This script combines alignment directories, such as exp/tri4a_ali, and
# validates matching of the utterances and alignments after combining.

# Begin configuration section.
cmd=run.pl
nj=4
combine_lat=true
combine_ali=true
tolerance=10
# End configuration section.
echo "$0 $@"  # Print the command line for logging.

[[ -f path.sh ]] && . ./path.sh
. parse_options.sh || exit 1

export LC_ALL=C

if [[ $# -lt 3 ]]; then
  cat >&2 <<EOF
Usage: $0 [options] <data> <dest-dir> <src-dir1> <src-dir2> ...
 e.g.: $0 --nj 32 data/train exp/tri3_ali_combined exp/tri3_ali_1 exp_tri3_ali_2
Options:
 --nj <nj>              # number of jobs to split combined archives [4]
 --combine_ali false    # merge ali.*.gz if present [true]
 --combine_lat false    # merge lat.*.gz if present [true]
 --tolerance <int,%>    # maximum percentage of missing alignments or lattices
                        # w.r.t. total utterances in <data> before error is
                        # reported [10]

The script checks that certain important files are present and compatible in all
source directories (phones.txt, tree); other are copied from the first source
(cmvn_opts, final.mdl) without much checking.

Both --combine_ali and --combine_lat are true by default, but the script
proceeds with a warning if directories do not contain either alignments or
alignment lattices. Check for files ali.1.gz and/or lat.1.gz in the <dest-dir>
after the script completes if additional programmatic check is required.
EOF
  exit 1;
fi

if [[ ! $combine_lat && ! $combine_ali ]]; then
  echo "$0: at least one of --combine_lat and --combine_ali must be true"
  exit 1
fi

data=$1
dest=$2
shift 2
first_src=$1

do_ali=$combine_ali
do_lat=$combine_lat

# Check if alignments and/or lattices are present. Since we combine both,
# whichever present, issue a warning only. Also verify that the target is
# different from any source; we cannot combine in-place, and a lot of damage
# could result.
for src in $@; do
  if [[ "$(cd 2>/dev/null -P -- "$src" && pwd)" = \
        "$(cd 2>/dev/null -P -- "$dest" && pwd)" ]]; then
    echo "$0: error: Source $src is same as target $dest."
    exit 1
  fi
  if $do_ali && [[ ! -f $src/ali.1.gz ]]; then
    echo "$0: warning: Alignments (ali.*.gz) are not present in $src, not" \
         "combining. Consider '--combine_ali false' to suppress this warning."
    do_ali=false
  fi
  if $do_lat && [[ ! -f $src/lat.1.gz ]]; then
    echo "$0: warning: Alignment lattices (lat.*.gz) are not present in $src,"\
      "not combining. Consider '--combine_lat false' to suppress this warning."
    do_lat=false
  fi
done

if ! $do_ali && ! $do_lat; then
  echo "$0: error: Cannot combine directories."
  exit 1
fi

# Verify that required files are present in the first directory.
for f in cmvn_opts final.mdl num_jobs phones.txt tree; do
  if [ ! -f $first_src/$f ]; then
    echo "$0: error: Required source file $first_src/$f is missing."
    exit 1
  fi
done

# Verify that phones and trees are compatible in all directories, and than
# num_jobs files are present, too.
for src in $@; do
  if [[ $src != $first_src ]]; then
    if [[ ! -f $src/num_jobs ]]; then
      echo "$0: error: Required source file $src/num_jobs is missing."
      exit 1
    fi
    if ! cmp -s $first_src/tree $src/tree; then
      echo "$0: error: tree $src/tree is either missing or not the" \
           "same as $first_src/tree."
      exit 1
    fi
    if [[ ! -f $src/phones.txt ]]; then
      echo "$0: error: Required source file $src/phones.txt is missing."
      exit 1
    fi
    utils/lang/check_phones_compatible.sh $first_src/phones.txt \
                                          $src/phones.txt || exit 1
  fi
done

# All checks passed, ok to prepare directory. Copy model and other files from
# the first source, as they either checked to be compatible, or we do not care
# if they are.
mkdir -p $dest || exit 1
rm -f $dest/{cmvn_opts,final.mdl,num_jobs,phones.txt,tree}
$do_ali && rm -f $dest/ali.*.{gz,scp}
$do_lat && rm -f $dest/lat.*.{gz,scp}
cp $first_src/{cmvn_opts,final.mdl,phones.txt,tree} $dest/ || exit 1
cp $first_src/frame_subsampling_factor $dest/ 2>/dev/null  # If present.
echo $nj > $dest/num_jobs || exit 1

# Make temporary directory, delete on signal, but not on 'exit 1'.
temp_dir=$(mktemp -d $dest/temp.XXXXXX) || exit 1
cleanup() { rm -rf "$temp_dir"; }
trap cleanup HUP INT TERM
echo "$0: note: Temporary directory $temp_dir will not be deleted in case of" \
     "script failure, so you could examine it for troubleshooting."


# This function may be called twice, once to combine alignments and the second
# time to combine lattices. The two invocations are as follows:
#   do_combine ali alignments copy-int-vector $@
#   do_combine lat lattices   lattice-copy $@
# where 'ali'/'lat' is a prefix to archive name, 'alignments'/'lattices' go into
# log messages and logfile names, and 'copy-int-vector'/'lattice-copy' is the
# program used to copy corresponding objects.
do_combine() {
  local ark=$1 entities=$2 copy_program=$3
  shift 3

  echo "$0: Gathering $entities from each source directory."
  # Assign all source gzipped archive names to an exported variable, one each
  # per source directory, so that we can copy archives in a job per source.
  src_id=0
  new_id=0
  for src in $@; do
    src_id=$((src_id + 1))
    nj_src=$(cat $src/num_jobs) || exit 1
    # Create and export variable src_arcs_${src_id} for the job runner.
    # Each numbered variable will contain the list of archives, e. g.:
    # src_arcs_1="exp/tri3_ali/ali.1.gz exp/tri3_ali/ali.1.gz ..."
    # ('printf' repeats its format as long as there are more arguments).
    for src_nj_id in $(seq $nj_src); do
      new_id=$((new_id + 1))
      printf "$src/$ark.%d.gz" $src_nj_id > $temp_dir/src_arks.${new_id}
    done
  done

  # Gather archives in parallel jobs.
  $cmd JOB=1:$new_id $dest/log/gather_$entities.JOB.log \
    $copy_program \
      "ark:gunzip -c \$(cat $temp_dir/src_arks.JOB) |" \
      "ark,scp:$temp_dir/$ark.JOB.ark,$temp_dir/$ark.JOB.scp" || exit 1

  # Merge (presumed already sorted) scp's into a single script.
  sort -m $temp_dir/$ark.*.scp > $temp_dir/$ark.scp || exit 1

  inputs=$(for n in `seq $nj`; do echo $temp_dir/$ark.$n.scp; done)
  utils/split_scp.pl --utt2spk=$data/utt2spk $temp_dir/$ark.scp $inputs

  echo "$0: Splitting combined $entities into $nj archives on speaker boundary."
  $cmd JOB=1:$nj $dest/log/chop_combined_$entities.JOB.log \
    $copy_program \
      "scp:$temp_dir/$ark.JOB.scp" \
      "ark:| gzip -c > $dest/$ark.JOB.gz" || exit 1

  # Get some interesting stats, and signal an error if error threshold exceeded.
  n_utt=$(wc -l <$data/utt2spk)
  n_ali=$(wc -l <$temp_dir/$ark.scp)
  n_ali_no_utt=$(join -j1 -v2 $data/utt2spk $temp_dir/$ark.scp | wc -l)
  n_utt_no_ali=$(join -j1 -v1 $data/utt2spk $temp_dir/$ark.scp | wc -l)
  n_utt_no_ali_pct=$(perl -e "print int($n_utt_no_ali/$n_utt * 100 + .5);")
  echo "$0: Combined $n_ali $entities for $n_utt utterances." \
       "There were $n_utt_no_ali utterances (${n_utt_no_ali_pct}%) without" \
       "$entities, and $n_ali_no_utt $entities not matching any utterance."

  if (( $n_utt_no_ali_pct >= $tolerance )); then
    echo "$0: error: Percentage of utterances missing $entities," \
         "${n_utt_no_ali_pct}%, is at or above error tolerance ${tolerance}%."
    exit 1
  fi

  return 0
}

# Do the actual combining. Do not check returned exit code, as
# the function always calls 'exit 1' on failure.
$do_ali && do_combine ali 'alignments' copy-int-vector "$@"
$do_lat && do_combine lat 'lattices' lattice-copy "$@"

# Delete the temporary directory on success.
cleanup

what=
$do_ali && what+='alignments '
$do_ali && $do_lat && what+='and '
$do_lat && what+='lattices '
echo "$0: Stored combined ${what}in $dest"  # No period, interferes with
                                            # copy/paste from tty emulator.
exit 0
