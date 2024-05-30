#!/usr/bin/perl
use warnings; #sed replacement for -w perl parameter
# Copyright 2017   David Snyder
# Apache 2.0
#
# Prepares the telephone portion of Mixer 6 (LDC2013S03).

if (@ARGV != 2) {
  print STDERR "Usage: $0 <path-to-LDC2013S03> <path-to-output>\n";
  print STDERR "e.g. $0 /export/corpora5/LDC/LDC2013S03 data/\n";
  exit(1);
}
($db_base, $out_dir) = @ARGV;

if (! -d "$db_base/mx6_speech/data/ulaw_sphere/") {
  print STDERR "Directory $db_base/mx6_speech/data/ulaw_sphere/ doesn't exist\n";
  exit(1);
}

$out_dir = "$out_dir/mx6_calls";

$tmp_dir = "$out_dir/tmp";
if (system("mkdir -p $tmp_dir") != 0) {
  die "Error making directory $tmp_dir";
}

if (system("mkdir -p $out_dir") != 0) {
  print STDERR "Error making directory $out_dir\n";
  exit(1);
}

%call2sph = ();
open(SUBJECTS, "<$db_base/mx6_speech/docs/mx6_subjs.csv") || die "cannot open $$db_base/mx6_speech/docs/mx6_subjs.csv";
open(SPKR, ">$out_dir/utt2spk") || die "Could not open the output file $out_dir/utt2spk";
open(GNDR, ">$out_dir/spk2gender") || die "Could not open the output file $out_dir/spk2gender";
open(WAV, ">$out_dir/wav.scp") || die "Could not open the output file $out_dir/wav.scp";
open(META, "<$db_base/mx6_speech/docs/mx6_calls.csv") || die "cannot open $db_base/mx6_speech/docs/mx6_calls.csv";

if (system("find $db_base/mx6_speech/data/ulaw_sphere/ -name '*.sph' > $tmp_dir/sph.list") != 0) {
  die "Error getting list of sph files";
}

open(SPHLIST, "<$tmp_dir/sph.list") or die "cannot open wav list";

while(<SPHLIST>) {
  chomp;
  $sph = $_;
  @toks = split("/",$sph);
  $sph_id = (split("[./]",$toks[$#toks]))[0];
  $call_id = (split("_", $sph_id))[2];
  $call2sph[$call_id] = $sph;
}

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
  $call_id = $toks[0];
  ($call_date, $call_time) = split(/_/, $toks[1]);
  $sid_A = $toks[4];
  $sid_B = $toks[12];
  if (-f $call2sph[$call_id]) {
    $utt_A = "${sid_A}_MX6_${call_id}_A";
    $utt_B = "${sid_B}_MX6_${call_id}_B";
    print SPKR "${utt_A} $sid_A\n";
    print SPKR "${utt_B} $sid_B\n";
    print WAV "${utt_A} sph2pipe -f wav -p -c 1 $call2sph[$call_id] |\n";
    print WAV "${utt_B} sph2pipe -f wav -p -c 2 $call2sph[$call_id] |\n";
    $num_good_files++;
  } else {
    print STDERR "Sphere file for $call_id doesn't exist\n";
    $num_bad_files++;
  }
}

print STDERR "Processed $num_good_files utterances; $num_bad_files had missing sphere data.\n";

close(SPHLIST) || die;
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
