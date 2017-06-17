#!/bin/bash

# Copyright 2014  Guoguo Chen
# Apache 2.0

# Begin configuration section.
seg_length=30
min_seg_length=10
overlap_length=5
# End configuration section.

echo "$0 $@"

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -ne 2 ]; then
  echo "This script truncates the long audio into smaller overlapping segments"
  echo ""
  echo "Usage: $0 [options] <input-dir> <output-dir>"
  echo " e.g.: $0 data/train_si284_long data/train_si284_split"
  echo ""
  echo "Options:"
  echo "    --min-seg-length        # minimal segment length"
  echo "    --seg-length            # length of segments in seconds."
  echo "    --overlap-length        # length of overlap in seconds."
  exit 1;
fi

input_dir=$1
output_dir=$2

for f in spk2utt text utt2spk wav.scp; do
  [ ! -f $input_dir/$f ] && echo "$0: no such file $input_dir/$f" && exit 1;
done

[ ! $seg_length -gt $overlap_length ] \
  && echo "$0: --seg-length should be longer than --overlap-length." && exit 1;

# Checks if sox is on the path.
sox=`which sox`
[ $? -ne 0 ] && echo "$0: sox command not found." && exit 1;
sph2pipe=$KALDI_ROOT/tools/sph2pipe_v2.5/sph2pipe
[ ! -x $sph2pipe ] && echo "$0: sph2pipe command not found." && exit 1;

mkdir -p $output_dir
cp -f $input_dir/spk2gender $output_dir/spk2gender 2>/dev/null
cp -f $input_dir/text $output_dir/text.orig
cp -f $input_dir/wav.scp $output_dir/wav.scp

# We assume the audio length in header is correct and get it from there. It is
# a little bit annoying that old version of sox does not support the following:
#   $audio_cmd | sox --i -D
# we have to put it in the following format for the old versions:
#   $sox --i -D "|$audio_cmd"
# Another way is to count all the samples to get the duration, but it takes
# longer time, so we do not use it here.. The command is:
#   $audio_cmd | sox -t wav - -n stat | grep -P "^Length" | awk '{print $1;}'
#
# Note: in the wsj example the process takes couple of minutes because of the
#       audio file concatenation; in a real case it should be much faster since
#       it just reads the header.
cat $output_dir/wav.scp | perl -e '
  $no_orig_seg = "false";       # Original segment file may or may not exist.
  ($u2s_in, $u2s_out, $seg_in,
   $seg_out, $orig2utt, $sox, $slen, $mslen, $olen) = @ARGV;
  open(UI, "<$u2s_in") || die "Error: fail to open $u2s_in\n";
  open(UO, ">$u2s_out") || die "Error: fail to open $u2s_out\n";
  open(SI, "<$seg_in") || ($no_orig_seg = "true");
  open(SO, ">$seg_out") || die "Error: fail to open $seg_out\n";
  open(UMAP, ">$orig2utt") || die "Error: fail to open $orig2utt\n";
  # If the original segment file exists, we have to work out the segment
  # duration from the segment file. Otherwise we work that out from the wav.scp
  # file.
  if ($no_orig_seg eq "false") {
    while (<SI>) {
      chomp;
      @col = split;
      @col == 4 || die "Error: bad line $_\n";
      ($seg_id, $wav_id, $seg_start, $seg_end) = @col;
      $seg2wav{$seg_id} = $wav_id;
      $seg_start{$seg_id} = $seg_start;
      $seg_end{$seg_id} = $seg_end;
    }
  } else {
    while (<STDIN>) {
      chomp;
      @col = split;
      @col >= 2 || "bad line $_\n";
      if ((@col > 2) &&  ($col[-1] eq "|")) {
        $wav_id = shift @col; pop @col;
        $audio_cmd = join(" ", @col);
        $duration = `$sox --i -D '\''|$audio_cmd'\''`;
      } else {
        @col == 2 || die "Error: bad line $_\n in wav.scp";
        $wav_id = $col[0];
        $audio_file = $col[1];
        $duration = `$sox --i -D $audio_file`;
      }
      chomp($duration);
      $seg2wav{$wav_id} = $wav_id;
      $seg_start{$wav_id} = 0;
      $seg_end{$wav_id} = $duration;
    }
  }
  while (<UI>) {
    chomp;
    @col = split;
    @col == 2 || die "Error: bad line $_\n";
    $utt2spk{$col[0]} = $col[1];
  }
  foreach $seg (sort keys %seg2wav) {
    $index = 0;
    $step = $slen - $olen;
    print UMAP "$seg";
    while ($seg_start{$seg} + $index * $step < $seg_end{$seg}) {
      $new_seg = $seg . "_" . sprintf("%05d", $index);
      $start = $seg_start{$seg} + $index * $step;
      $end = $start + $slen;
      defined($utt2spk{$seg}) || die "Error: speaker not found for $seg\n";
      print UO "$new_seg $utt2spk{$seg}\n";
      print UMAP " $new_seg"; 
      $index += 1;
      if ($end - $olen + $mslen >= $seg_end{$seg}) {
        # last segment will have at least $mslen seconds.
        $end = $seg_end{$seg};
        print SO "$new_seg $seg2wav{$seg} $start $end\n";
        last;
      } else {
        print SO "$new_seg $seg2wav{$seg} $start $end\n";
      }
    }
    print UMAP "\n";
  }' $input_dir/utt2spk $output_dir/utt2spk \
    $input_dir/segments $output_dir/segments $output_dir/orig2utt \
    $sox $seg_length $min_seg_length $overlap_length

# CAVEAT: We are not dealing with channels here. Each channel should have a
# unique file name in wav.scp.
paste -d ' ' <(cut -d ' ' -f 1 $output_dir/wav.scp) \
  <(cut -d ' ' -f 1 $output_dir/wav.scp) | awk '{print $1" "$2" A";}' \
  > $output_dir/reco2file_and_channel

utils/fix_data_dir.sh $output_dir

exit 0;
