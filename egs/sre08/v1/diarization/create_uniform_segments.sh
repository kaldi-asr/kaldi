#!/bin/bash
# Copyright 2015  Vimal Manohar
# Apache 2.0.

set -e 
set -o pipefail

# Begin configuration section.
cmd=run.pl
nj=4
window_length=600     # 10 minutes
frame_shift=0.01

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
  echo "Usage: create_uniform_segments.sh <in-data> <tmp-dir> <out-data>"
  echo " e.g.: diarization/train_vad_gmm_ntu.sh data/dev exp/uniform_segment_dev data/dev_uniform"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  exit 1;
fi

data=$1
tmpdir=$2
dir=$3

mkdir -p $tmpdir
mkdir -p $dir

utils/copy_data_dir.sh $data $dir
for f in feats.scp segments spk2utt utt2spk vad.scp cmvn.scp spk2gender text; do 
  [ -f $dir/$f ] && rm $dir/$f
done
rm -rf $dir/split* $dir/.backup

awk '{print $1" "$1}' $dir/wav.scp > $dir/utt2spk || exit 1
utils/utt2spk_to_spk2utt.pl $dir/utt2spk > $dir/spk2utt || exit 1

utils/fix_data_dir.sh $dir

utils/split_data.sh $dir $nj

$cmd JOB=1:$nj $tmpdir/log/get_wav_lengths.JOB.log \
  wav-to-duration scp:$dir/split$nj/JOB/wav.scp \
  ark,t:$tmpdir/wav_lengths.JOB.txt || exit 1

for n in `seq $nj`; do
  cat $tmpdir/wav_lengths.$n.txt
done | perl -e '
$frame_shift = $ARGV[0];
$window_length = $ARGV[1];
while (<STDIN>) {
  chomp;
  @F = split;
  $file_id = $F[0];
  $duration = $F[1];

  $num_chunks = int($duration / $window_length + 0.99);
  $this_window_length = $duration / $num_chunks;

  for ($i = 0; $i < $num_chunks; $i++) {
    $start_time = $i * $this_window_length;
    $end_time = ($i+1) * $this_window_length;
    $end_time = $duration if ($end_time > $duration);

    $start_str = sprintf("%06d", $start_time);
    $end_str = sprintf("%06d", $end_time);

    printf STDOUT ("$file_id-$start_str-$end_str $file_id %.2f %.2f\n", $start_time, $end_time);
  }
}' $frame_shift $window_length > $dir/segments || exit 1

[ ! -s $dir/segments ] && echo "$0: No segments in $dir/segments" && exit 1

awk '{print $1" "$2}' $dir/segments > $dir/utt2spk || exit 1
utils/utt2spk_to_spk2utt.pl $dir/utt2spk > $dir/spk2utt || exit 1

utils/fix_data_dir.sh $dir

