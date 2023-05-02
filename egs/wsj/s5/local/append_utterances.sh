#!/usr/bin/env bash

# Copyright 2014  Guoguo Chen
# Apache 2.0

# Begin configuration section.
pad_silence=0.5
# End configuration section.

echo "$0 $@"

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -ne 2 ]; then
  echo "Usage: $0 [options] <input-dir> <output-dir>"
  echo "Options:"
  echo "    --pad-silence           # silence to be padded between utterances"
  exit 1;
fi

input_dir=$1
output_dir=$2

for f in spk2gender spk2utt text utt2spk wav.scp; do
  [ ! -f $input_dir/$f ] && echo "$0: no such file $input_dir/$f" && exit 1;
done

# Checks if sox is on the path.
sox=`which sox`
[ $? -ne 0 ] && "sox: command not found." && exit 1;
sph2pipe=$KALDI_ROOT/tools/sph2pipe_v2.5/sph2pipe
[ ! -x $sph2pipe ] && "sph2pipe: command not found." && exit 1;

mkdir -p $output_dir
cp -f $input_dir/spk2gender $output_dir/spk2gender

# Creates a silence wav file. We create this actual sil.wav file instead of
# using sox's padding because this way sox can properly pipe the length in the
# header file. Otherwise sox will have to "count" all the samples and then
# update the header, which is not proper in pipe.
mkdir -p $output_dir/.tmp
$sox -n -r 16000 -b 16 $output_dir/.tmp/sil.wav trim 0.0 $pad_silence

cat $input_dir/spk2utt | perl -e '
  ($text_in, $wav_in, $text_out, $wav_out, $sox, $sph2pipe, $sil_wav) = @ARGV;
  open(TI, "<$text_in") || die "Error: fail to open $text_in\n";
  open(TO, ">$text_out") || die "Error: fail to open $text_out\n";
  open(WI, "<$wav_in") || die "Error: fail to open $wav_in\n";
  open(WO, ">$wav_out") || die "Error: fail to open $wav_out\n";
  while (<STDIN>) {
    chomp;
    my @col = split;  # We need to add "my" since we use reference below.
    @col >= 2 || "bad line $_\n";
    $spk = shift @col;
    $spk2utt{$spk} = \@col;
  }
  while (<TI>) {
    chomp;
    @col = split;
    @col >= 2 || die "Error: bad line $_\n";
    $utt = shift @col;
    $text{$utt} = join(" ", @col);
  }
  while (<WI>) {
    chomp;
    @col = split;
    @col >= 2 || die "Error: bad line $_\n";
    $wav{$col[0]} = $col[4];
  }
  foreach $spk (keys %spk2utt) {
    @utts = @{$spk2utt{$spk}};
    # print $utts[0] . "\n";
    $text_line = "";
    $wav_line = " $sox";
    foreach $utt (@utts) {
      $text_line .=  " " . $text{$utt};
      $wav_line .= " \"| $sph2pipe -f wav $wav{$utt}\"";  # speech
      $wav_line .= " $sil_wav";                           # silence
    }
    $text_line = $spk . $text_line . "\n";
    $wav_line = $spk . $wav_line . " -t wav - |\n";
    print TO $text_line;
    print WO $wav_line;
  }' $input_dir/text $input_dir/wav.scp $output_dir/text \
    $output_dir/wav.scp $sox $sph2pipe $output_dir/.tmp/sil.wav

cat $input_dir/spk2utt | awk '{print $1" "$1;}' > $output_dir/spk2utt
utils/spk2utt_to_utt2spk.pl $output_dir/spk2utt > $output_dir/utt2spk

utils/fix_data_dir.sh $output_dir

exit 0;
