#!/usr/bin/env bash
#
# Copyright Johns Hopkins University (Author: Daniel Povey) 2016.  Apache 2.0.

# This script does the same type of diagnostics as analyze_alignments.sh, except
# it starts from lattices (so it has to convert the lattices to alignments
# first).

# begin configuration section.
iter=final
cmd=run.pl
acwt=0.1
#end configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -ne 2 ]; then
  echo "Usage: $0 [options] (<lang-dir>|<graph-dir>) <decode-dir>"
  echo " Options:"
  echo "    --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes."
  echo "    --acwt <acoustic-scale>         # Acoustic scale for getting best-path (default: 0.1)"
  echo "e.g.:"
  echo "$0 data/lang exp/tri4b/decode_dev"
  echo "This script writes some diagnostics to <decode-dir>/log/alignments.log"
  exit 1;
fi

lang=$1
dir=$2

model=$dir/../${iter}.mdl

for f in $lang/words.txt $model $dir/lat.1.gz $dir/num_jobs; do
  [ ! -f $f ] && echo "$0: expecting file $f to exist" && exit 1;
done

num_jobs=$(cat $dir/num_jobs) || exit 1

mkdir -p $dir/log

rm $dir/phone_stats.*.gz 2>/dev/null || true

# this writes two archives of depth_tmp and ali_tmp of (depth per frame, alignment per frame).
$cmd JOB=1:$num_jobs $dir/log/lattice_best_path.JOB.log \
  lattice-depth-per-frame "ark:gunzip -c $dir/lat.JOB.gz|" "ark,t:|gzip -c > $dir/depth_tmp.JOB.gz" ark:- \| \
  lattice-best-path --acoustic-scale=$acwt ark:- ark:/dev/null "ark,t:|gzip -c >$dir/ali_tmp.JOB.gz" || exit 1

$cmd JOB=1:$num_jobs $dir/log/get_lattice_stats.JOB.log \
  ali-to-phones --write-lengths=true "$model" "ark:gunzip -c $dir/ali_tmp.JOB.gz|" ark,t:- \| \
  perl -ne 'chomp;s/^\S+\s*//;@a=split /\s;\s/, $_;$count{"begin ".$a[$0]."\n"}++;
  if(@a>1){$count{"end ".$a[-1]."\n"}++;}for($i=0;$i<@a;$i++){$count{"all ".$a[$i]."\n"}++;}
  END{for $k (sort keys %count){print "$count{$k} $k"}}' \| \
  gzip -c '>' $dir/phone_stats.JOB.gz || exit 1

$cmd $dir/log/analyze_alignments.log \
  gunzip -c "$dir/phone_stats.*.gz" \| \
  steps/diagnostic/analyze_phone_length_stats.py $lang || exit 1

grep WARNING $dir/log/analyze_alignments.log
echo "$0: see stats in $dir/log/analyze_alignments.log"

$cmd $dir/log/dump_ali_frame.log \
  ali-to-phones --per-frame=true "$model" "ark:gunzip -c $dir/ali_tmp.*.gz|" "ark,t:|gzip -c >$dir/ali_frame_tmp.gz"

$cmd $dir/log/analyze_lattice_depth_stats.log \
  gunzip -c "$dir/depth_tmp.*.gz" \| \
  steps/diagnostic/analyze_lattice_depth_stats.py $lang "$dir/ali_frame_tmp.gz" || exit 1

grep Overall $dir/log/analyze_lattice_depth_stats.log
echo "$0: see stats in $dir/log/analyze_lattice_depth_stats.log"


rm $dir/phone_stats.*.gz
rm $dir/depth_tmp.*.gz
rm $dir/ali_frame_tmp.gz
rm $dir/ali_tmp.*.gz

exit 0
