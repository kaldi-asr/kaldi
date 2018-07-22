#!/usr/bin/perl
#
# Copyright 2015   David Snyder
#           2017   Radboud University (Author: Emre Yilmaz)
# Apache 2.0.
# Usage: make_fame_test_year.pl corpus/SV/ data/ complete 3sec eval 1t3.

if (@ARGV != 5) {
  print STDERR "Usage: $0 <path-to-FAME corpus> <path-to-output> <task-name> <subtask-name> <dataset-name> <age-category>\n";
  print STDERR "e.g. $0 corpus/SV/ data/ complete 3sec eval\n";
  exit(1);
}

($db_base, $out_base_dir, $task, $subtask, $sets, $year) = @ARGV;
$out_dir = "$out_base_dir/fame_${task}_${subtask}_${sets}${year}_test";

$tmp_dir = "$out_dir/tmp";
if (system("mkdir -p $tmp_dir") != 0) {
  die "Error making directory $tmp_dir"; 
}

open(IN_TRIALS, "<", "$db_base/docs/$task/${task}_${subtask}_${sets}_trials${year}_key") or die "cannot open trials list";
open(OUT_TRIALS, ">", "$out_dir/trials") or die "cannot open trials list";
%trials = ();
while(<IN_TRIALS>) {
  chomp;
  ($spkr,$utt,$side,$is_target) = split(",", $_);
  $side = uc $side;
  $key = "${spkr} ${utt}_${side}"; # Just keep track of the spkr-utterance pairs we want.
  $trials{$key} = 1; # Just keep track of the spkr-utterance pairs we want.
  print OUT_TRIALS "$spkr ${utt}_${side} $is_target\n";
}

close(OUT_TRIALS) || die;
close(IN_TRIALS) || die;

open(WAVLIST, "<", "$db_base/docs/$task/${task}_${subtask}_${sets}_trials${year}") or die "cannot open wav list";
open(GNDR,">", "$out_dir/spk2gender") or die "Could not open the output file $out_dir/spk2gender";
open(SPKR,">", "$out_dir/utt2spk") or die "Could not open the output file $out_dir/utt2spk";
open(WAV,">", "$out_dir/wav.scp") or die "Could not open the output file $out_dir/wav.scp";

%spk2gender = ();
%utts = ();
while(<WAVLIST>) {
  chomp;
  $sph = $_;
  ($spkr, $gender, $wav_and_side) = split(" ", $sph);
  ($wav, $side) = split(":", $wav_and_side);
  $wav = "${db_base}/data/${task}/${sets}/${subtask}/${wav}";
  @A = split("/", $wav);
  $basename = $A[$#A];
  $raw_basename = $basename;
  $raw_basename =~ s/\.wav$// || die "bad basename $basename";
  $uttId = $raw_basename . "_" . $side;
  $key = "${spkr} ${uttId}";
  if ( (not exists($trials{"${spkr} ${uttId}"}) ) or exists($utts{$uttId})  ) {
    next;
  }
  $utts{$uttId} = 1;
  if ($side eq "A") {
    $channel = 1;
  } elsif ($side eq "B") {
    $channel = 2;
  } else {
    die "unknown channel $side\n";
  }
  print WAV "$uttId"," $wav\n";
  print SPKR "$uttId $uttId\n";
  print GNDR "$uttId $gender\n";
  $spk2gender{$spkr} = $gender;
}

close(SPKR) || die;
close(WAV) || die;
close(WAVLIST) || die;
close(GNDR) || die;

if (system(
  "utils/utt2spk_to_spk2utt.pl $out_dir/utt2spk >$out_dir/spk2utt") != 0) {
  die "Error creating spk2utt file in directory $out_dir";
}
system("utils/fix_data_dir.sh $out_dir");
if (system("utils/validate_data_dir.sh --no-text --no-feats $out_dir") != 0) {
  die "Error validating directory $out_dir";
}
