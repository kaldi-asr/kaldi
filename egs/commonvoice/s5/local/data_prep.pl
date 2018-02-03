#!/usr/bin/perl
#
# Copyright 2017   Ewald Enzinger
# Apache 2.0
#
# Usage: data_prep.pl /export/data/cv_corpus_v1/cv-valid-train valid_train

if (@ARGV != 3) {
  print STDERR "Usage: $0 <path-to-commonvoice-corpus> <dataset> <valid-train|valid-dev|valid-test>\n";
  print STDERR "e.g. $0 /export/data/cv_corpus_v1 cv-valid-train valid-train\n";
  exit(1);
}

($db_base, $dataset, $out_dir) = @ARGV;
mkdir data unless -d data;
mkdir $out_dir unless -d $out_dir;

open(CSV, "<", "$db_base/$dataset.csv") or die "cannot open dataset CSV file";
open(SPKR,">", "$out_dir/utt2spk") or die "Could not open the output file $out_dir/utt2spk";
open(GNDR,">", "$out_dir/utt2gender") or die "Could not open the output file $out_dir/utt2gender";
open(TEXT,">", "$out_dir/text") or die "Could not open the output file $out_dir/text";
open(WAV,">", "$out_dir/wav.scp") or die "Could not open the output file $out_dir/wav.scp";
my $header = <CSV>;
while(<CSV>) {
  chomp;
  ($filepath, $text, $upvotes, $downvotes, $age, $gender, $accent, $duration) = split(",", $_);
  if ("$gender" eq "female") {
    $gender = "f";
  } else {
    # Use male as default if not provided (no reason, just adopting the same default as in voxforge)
    $gender = "m";
  }
  $uttId = $filepath;
  $uttId =~ s/\.mp3//g;
  $uttId =~ tr/\//-/;
  # No speaker information is provided, so we treat each utterance as coming from a different speaker
  $spkr = $uttId;
  $text =~ s/ said 'eat when/ said eat when/g;
  $text =~ s/'and this is what your son said'/and this is what your son said/g;
  $text =~ s/^'m /i'm /g;
  $text =~ s/'mummy'/mummy/g;
  $text =~ s/'poppy'/poppy/g;
  $text =~ s/'every/every/g;
  $text =~ s/'super fun playground'/super fun playground/g;
  $text =~ s/'under construction'/under construction/g;
  $text =~ tr/a-z/A-Z/;
  print TEXT "$uttId"," ","$text","\n";
  print GNDR "$uttId"," ","$gender","\n";
  print WAV "$uttId"," sox $db_base/$filepath -t wav -r 16k -b 16 -e signed - |\n";
  print SPKR "$uttId"," $spkr","\n";
}
close(SPKR) || die;
close(TEXT) || die;
close(WAV) || die;
close(GNDR) || die;
close(WAVLIST);

if (system(
  "utils/utt2spk_to_spk2utt.pl $out_dir/utt2spk >$out_dir/spk2utt") != 0) {
  die "Error creating spk2utt file in directory $out_dir";
}
system("env LC_COLLATE=C utils/fix_data_dir.sh $out_dir");
if (system("env LC_COLLATE=C utils/validate_data_dir.sh --no-feats $out_dir") != 0) {
  die "Error validating directory $out_dir";
}
