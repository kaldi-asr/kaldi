#!/usr/bin/perl
#
# Copyright 2015   David Snyder
#           2017   Radboud University (Author: Emre Yilmaz)
# Apache 2.0.
# Usage: make_fame_train_year.pl corpus/SV/ data/ complete 3sec eval 1t3.

if (@ARGV != 5) {
  print STDERR "Usage: $0 <path-to-FAME corpus> <path-to-output> <task-name> <subtask-name> <dataset-name> <age-category>\n";
  print STDERR "e.g. $0 corpus/SV/ data/ complete 3sec eval\n";
  exit(1);
}

($db_base, $out_base_dir, $task, $subtask, $sets, $year) = @ARGV;
$out_dir = "$out_base_dir/fame_${task}_${subtask}_${sets}${year}_enroll";

$tmp_dir = "$out_dir/tmp";
if (system("mkdir -p $tmp_dir") != 0) {
  die "Error making directory $tmp_dir"; 
}

open(WAVLIST, "<", "$db_base/docs/${task}/${task}_${subtask}_${sets}_enroll${year}") or die "cannot open wav list";
open(SPKR,">", "$out_dir/utt2spk") or die "Could not open the output file $out_dir/utt2spk";
open(GNDR,">", "$out_dir/spk2gender") or die "Could not open the output file $out_dir/spk2gender";
open(WAV,">", "$out_dir/wav.scp") or die "Could not open the output file $out_dir/wav.scp";

while(<WAVLIST>) {
  chomp;
  $sph = $_;
  ($spkr, $gender, $wav_and_side) = split(" ", $sph);
  ($wav, $side) = split(":", $wav_and_side);
  @A = split("/", $wav);
  $wav = "${db_base}/data/${task}/${sets}/${subtask}/${wav}";
  $basename = $A[$#A];
  $raw_basename = $basename;
  $raw_basename =~ s/\.wav$// || die "bad basename $basename";
  $uttId = $raw_basename . "_" . $side; # prefix spkr-id to utt-id to ensure sorted order.
  if ($side eq "A") {
    $channel = 1;
  } elsif ($side eq "B") {
    $channel = 2;
  } else {
    die "unknown channel $side\n";
  }
  print GNDR "$spkr $gender\n";
  print WAV "$uttId"," $wav", "\n";
  print SPKR "$uttId"," $spkr","\n";
}
close(GNDR) || die;
close(SPKR) || die;
close(WAV) || die;
close(WAVLIST) || die;
if (system(
  "utils/utt2spk_to_spk2utt.pl $out_dir/utt2spk >$out_dir/spk2utt") != 0) {
  die "Error creating spk2utt file in directory $out_dir";
}
system("utils/fix_data_dir.sh $out_dir");
if (system("utils/validate_data_dir.sh --no-text --no-feats $out_dir") != 0) {
  die "Error validating directory $out_dir";
}
