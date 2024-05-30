#!/usr/bin/perl
use warnings; #sed replacement for -w perl parameter
# Copyright 2017   David Snyder
# Apache 2.0
# Prepares Mixer 6 (LDC2013S03) speech from a specified microphone and
# downsamples it to 8k.

if (@ARGV != 3) {
  print STDERR "Usage: $0 <path-to-LDC2013S03> <channel> <path-to-output>\n";
  print STDERR "e.g. $0 /export/corpora5/LDC/LDC2013S03 02 data/\n";
  exit(1);
}
($db_base, $ch, $out_dir) = @ARGV;

@bad_channels = ("01", "03", "14");
if (/$ch/i ~~ @bad_channels) {
  print STDERR "Bad channel $ch\n";
  exit(1);
}

if (! -d "$db_base/mx6_speech/data/pcm_flac/CH$ch/") {
  print STDERR "Directory $db_base/mx6_speech/data/pcm_flac/CH$ch/ doesn't exist\n";
  exit(1);
}

$out_dir = "$out_dir/mx6_mic_$ch";
if (system("mkdir -p $out_dir")) {
  print STDERR "Error making directory $out_dir\n";
  exit(1);
}

if (system("mkdir -p $out_dir") != 0) {
  print STDERR "Error making directory $out_dir\n";
  exit(1);
}

open(SUBJECTS, "<$db_base/mx6_speech/docs/mx6_subjs.csv") || die "cannot open $$db_base/mx6_speech/docs/mx6_subjs.csv";
open(SPKR, ">$out_dir/utt2spk") || die "Could not open the output file $out_dir/utt2spk";
open(GNDR, ">$out_dir/spk2gender") || die "Could not open the output file $out_dir/spk2gender";
open(WAV, ">$out_dir/wav.scp") || die "Could not open the output file $out_dir/wav.scp";
open(META, "<$db_base/mx6_speech/docs/mx6_ivcomponents.csv") || die "cannot open $db_base/mx6_speech/docs/mx6_ivcomponents.csv";

while (<SUBJECTS>) {
  chomp;
  $line = $_;
  @toks = split(",", $line);
  $spk = $toks[0];
  $gender = lc $toks[1];
  if ($gender eq "f" or $gender eq "m") {
    print GNDR "$spk $gender\n";
  }
}

$num_good_files = 0;
$num_bad_files = 0;
while (<META>) {
  chomp;
  $line = $_;
  @toks = split(",", $line);
  $flac = "$db_base/mx6_speech/data/pcm_flac/CH$ch/$toks[0]_CH$ch.flac";
  $t1 = $toks[7];
  $t2 = $toks[8];
  @toks2 = split(/_/, $toks[0]);
  $spk = $toks2[3];
  $utt = "${spk}_MX6_$toks2[0]_$toks2[1]_$ch";
  if (-f $flac) {
    print SPKR "${utt} $spk\n";
    print WAV "${utt} sox -t flac $flac -r 8k -t wav - trim $t1 =$t2 |\n";
    $num_good_files++;
  } else {
    print STDERR "File $flac doesn't exist\n";
    $num_bad_files++;
  }
}

print STDERR "Processed $num_good_files utterances; $num_bad_files had missing flac data.\n";

close(SUBJECTS) || die;
close(GNDR) || die;
close(SPKR) || die;
close(WAV) || die;
close(META) || die;

if (system(
  "utils/utt2spk_to_spk2utt.pl $out_dir/utt2spk >$out_dir/spk2utt") != 0) {
  die "Error creating spk2utt file in directory $out_dir";
}

system("utils/fix_data_dir.sh $out_dir");
if (system("utils/validate_data_dir.sh --no-text --no-feats $out_dir") != 0) {
  die "Error validating directory $out_dir";
}
