#!/usr/bin/perl
#
# Copyright 2018  Ewald Enzinger
#           2018  David Snyder
#
# Usage: make_voxceleb1.pl /export/voxceleb1 data/

if (@ARGV != 2) {
  print STDERR "Usage: $0 <path-to-voxceleb1> <path-to-data-dir>\n";
  print STDERR "e.g. $0 /export/voxceleb1 data/\n";
  exit(1);
}

($data_base, $out_dir) = @ARGV;
my $out_test_dir = "$out_dir/voxceleb1_test";
my $out_train_dir = "$out_dir/voxceleb1_train";

if (system("mkdir -p $out_test_dir") != 0) {
  die "Error making directory $out_test_dir";
}

if (system("mkdir -p $out_train_dir") != 0) {
  die "Error making directory $out_train_dir";
}

opendir my $dh, "$data_base/voxceleb1_wav" or die "Cannot open directory: $!";
my @spkr_dirs = grep {-d "$data_base/voxceleb1_wav/$_" && ! /^\.{1,2}$/} readdir($dh);
closedir $dh;

if (! -e "$data_base/voxceleb1_test.txt") {
  system("wget -O $data_base/voxceleb1_test.txt http://www.openslr.org/resources/49/voxceleb1_test.txt");
}

if (! -e "$data_base/vox1_meta.csv") {
  system("wget -O $data_base/vox1_meta.csv http://www.openslr.org/resources/49/vox1_meta.csv");
}

open(TRIAL_IN, "<", "$data_base/voxceleb1_test.txt") or die "Could not open the verification trials file $data_base/voxceleb1_test.txt";
open(META_IN, "<", "$data_base/vox1_meta.csv") or die "Could not open the meta data file $data_base/vox1_meta.csv";
open(SPKR_TEST, ">", "$out_test_dir/utt2spk") or die "Could not open the output file $out_test_dir/utt2spk";
open(WAV_TEST, ">", "$out_test_dir/wav.scp") or die "Could not open the output file $out_test_dir/wav.scp";
open(SPKR_TRAIN, ">", "$out_train_dir/utt2spk") or die "Could not open the output file $out_train_dir/utt2spk";
open(WAV_TRAIN, ">", "$out_train_dir/wav.scp") or die "Could not open the output file $out_train_dir/wav.scp";
open(TRIAL_OUT, ">", "$out_test_dir/trials") or die "Could not open the output file $out_test_dir/trials";

my %id2spkr = ();
while (<META_IN>) {
  chomp;
  my ($vox_id, $spkr_id, $gender, $nation, $set) = split;
  $id2spkr{$vox_id} = $spkr_id;
}

my $test_spkrs = ();
while (<TRIAL_IN>) {
  chomp;
  my ($tar_or_non, $path1, $path2) = split;

  # Create entry for left-hand side of trial
  my ($spkr_id, $filename) = split('/', $path1);
  my $rec_id = substr($filename, 0, 11);
  my $segment = substr($filename, 12, 7);
  my $utt_id1 = "$spkr_id-$rec_id-$segment";
  $test_spkrs{$spkr_id} = ();

  # Create entry for right-hand side of trial
  my ($spkr_id, $filename) = split('/', $path2);
  my $rec_id = substr($filename, 0, 11);
  my $segment = substr($filename, 12, 7);
  my $utt_id2 = "$spkr_id-$rec_id-$segment";
  $test_spkrs{$spkr_id} = ();

  my $target = "nontarget";
  if ($tar_or_non eq "1") {
    $target = "target";
  }
  print TRIAL_OUT "$utt_id1 $utt_id2 $target\n";
}

foreach (@spkr_dirs) {
  my $spkr_id = $_;
  my $new_spkr_id = $spkr_id;
  # If we're using a newer version of VoxCeleb1, we need to "deanonymize"
  # the speaker labels.
  if (exists $id2spkr{$spkr_id}) {
    $new_spkr_id = $id2spkr{$spkr_id};
  }
  opendir my $dh, "$data_base/voxceleb1_wav/$spkr_id/" or die "Cannot open directory: $!";
  my @files = map{s/\.[^.]+$//;$_}grep {/\.wav$/} readdir($dh);
  closedir $dh;
  foreach (@files) {
    my $filename = $_;
    my $rec_id = substr($filename, 0, 11);
    my $segment = substr($filename, 12, 7);
    my $wav = "$data_base/voxceleb1_wav/$spkr_id/$filename.wav";
    my $utt_id = "$new_spkr_id-$rec_id-$segment";
    if (exists $test_spkrs{$new_spkr_id}) {
      print WAV_TEST "$utt_id", " $wav", "\n";
      print SPKR_TEST "$utt_id", " $new_spkr_id", "\n";
    } else {
      print WAV_TRAIN "$utt_id", " $wav", "\n";
      print SPKR_TRAIN "$utt_id", " $new_spkr_id", "\n";
    }
  }
}

close(SPKR_TEST) or die;
close(WAV_TEST) or die;
close(SPKR_TRAIN) or die;
close(WAV_TRAIN) or die;
close(TRIAL_OUT) or die;
close(TRIAL_IN) or die;
close(META_IN) or die;

if (system(
  "utils/utt2spk_to_spk2utt.pl $out_test_dir/utt2spk >$out_test_dir/spk2utt") != 0) {
  die "Error creating spk2utt file in directory $out_test_dir";
}
system("env LC_COLLATE=C utils/fix_data_dir.sh $out_test_dir");
if (system("env LC_COLLATE=C utils/validate_data_dir.sh --no-text --no-feats $out_test_dir") != 0) {
  die "Error validating directory $out_test_dir";
}

if (system(
  "utils/utt2spk_to_spk2utt.pl $out_train_dir/utt2spk >$out_train_dir/spk2utt") != 0) {
  die "Error creating spk2utt file in directory $out_train_dir";
}
system("env LC_COLLATE=C utils/fix_data_dir.sh $out_train_dir");
if (system("env LC_COLLATE=C utils/validate_data_dir.sh --no-text --no-feats $out_train_dir") != 0) {
  die "Error validating directory $out_train_dir";
}
