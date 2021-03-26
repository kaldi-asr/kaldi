#!/usr/bin/perl
#
# Copyright 2018  Ewald Enzinger
#           2018  David Snyder
#           2019  Soonshin Seo
#
# Usage: make_voxceleb1_v2.pl /export/voxceleb1 dev data/dev
#
# The VoxCeleb1 corpus underwent several updates that changed the directory and speaker ID format.
# The script 'make_voxceleb1.pl' works for the oldest version of the corpus. 
# This script should be used if you've downloaded the corpus recently.

if (@ARGV != 3) {
  print STDERR "Usage: $0 <path-to-voxceleb1> <dataset> <path-to-data-dir>\n";
  print STDERR "e.g. $0 /export/voxceleb1 dev data/dev\n";
  exit(1);
}

($data_base, $dataset, $out_dir) = @ARGV;

if ("$dataset" ne "dev" && "$dataset" ne "test") {
  die "dataset parameter must be 'dev' or 'test'!";
}

if (system("mkdir -p $out_dir") != 0) {
  die "Error making directory $out_dir";
}
print "$data_base/$dataset/wav\n";
opendir my $dh, "$data_base/$dataset/wav" or die "Cannot open directory: $!";
my @spkr_dirs = grep {-d "$data_base/$dataset/wav/$_" && ! /^\.{1,2}$/} readdir($dh);
closedir $dh;

if ($dataset eq "dev"){
  open(SPKR_TRAIN, ">", "$out_dir/utt2spk") or die "could not open the output file $out_dir/utt2spk";
  open(WAV_TRAIN, ">", "$out_dir/wav.scp") or die "could not open the output file $out_dir/wav.scp";

  foreach (@spkr_dirs) {
    my $spkr_id = $_;
    opendir my $dh, "$data_base/$dataset/wav/$spkr_id/" or die "Cannot open directory: $!";
    my @rec_dirs = grep {-d "$data_base/$dataset/wav/$spkr_id/$_" && ! /^\.{1,2}$/} readdir($dh);
    closedir $dh;
    foreach (@rec_dirs) {
	  my $rec_id = $_;
	  opendir my $dh, "$data_base/$dataset/wav/$spkr_id/$rec_id/" or die "Cannot open directory: $!";
	  my @files = map{s/\.[^.]+$//;$_}grep {/\.wav$/} readdir($dh);
	  closedir $dh;
  	  foreach (@files) {
        my $name = $_;
        my $wav = "$data_base/$dataset/wav/$spkr_id/$rec_id/$name.wav";
        my $utt_id = "$spkr_id-$rec_id-$name";
        print WAV_TRAIN "$utt_id", " $wav", "\n";
        print SPKR_TRAIN "$utt_id", " $spkr_id", "\n";
      }
    }
  }
  close(SPKR_TRAIN) or die;
  close(WAV_TRAIN) or die;
}

if ($dataset eq "test"){
  if (! -e "$data_base/voxceleb1_test_v2.txt") {
    system("wget -O $data_base/voxceleb1_test_v2.txt http://www.openslr.org/resources/49/voxceleb1_test_v2.txt");
  }

  open(TRIAL_IN, "<", "$data_base/voxceleb1_test_v2.txt") or die "could not open the verification trials file $data_base/voxceleb1_test_v2.txt";
  open(TRIAL_OUT, ">", "$out_dir/trials") or die "Could not open the output file $out_test_dir/trials";
  open(SPKR_TEST, ">", "$out_dir/utt2spk") or die "could not open the output file $out_dir/utt2spk";
  open(WAV_TEST, ">", "$out_dir/wav.scp") or die "could not open the output file $out_dir/wav.scp";

  my $test_spkrs = ();
  while (<TRIAL_IN>) {
    chomp;
    my ($tar_or_non, $path1, $path2) = split;
    # Create entry for left-hand side of trial
    my ($spkr_id, $rec_id, $name) = split('/', $path1);
    my $utt_id1 = "$spkr_id-$rec_id-$name";
    $test_spkrs{$spkr_id} = ();

    # Create entry for right-hand side of trial
    my ($spkr_id, $rec_id, $name) = split('/', $path2);
    my $utt_id2 = "$spkr_id-$rec_id-$name";
    $test_spkrs{$spkr_id} = ();

    my $target = "nontarget";
    if ($tar_or_non eq "1") {
      $target = "target";
    }
    print TRIAL_OUT "$utt_id1 $utt_id2 $target\n";
  }

  foreach (@spkr_dirs) {
    my $spkr_id = $_;
    opendir my $dh, "$data_base/$dataset/wav/$spkr_id/" or die "Cannot open directory: $!";
    my @rec_dirs = grep {-d "$data_base/$dataset/wav/$spkr_id/$_" && ! /^\.{1,2}$/} readdir($dh);
    closedir $dh;
    foreach (@rec_dirs) {
	  my $rec_id = $_;
	  opendir my $dh, "$data_base/$dataset/wav/$spkr_id/$rec_id/" or die "Cannot open directory: $!";
	  my @files = map{s/\.[^.]+$//;$_}grep {/\.wav$/} readdir($dh);
	  closedir $dh;
  	  foreach (@files) {
        my $name = $_;
        my $wav = "$data_base/$dataset/wav/$spkr_id/$rec_id/$name.wav";
        my $utt_id = "$spkr_id-$rec_id-$name";
        print WAV_TEST "$utt_id", " $wav", "\n";
        print SPKR_TEST "$utt_id", " $spkr_id", "\n";
      }
    }
  }
  close(SPKR_TEST) or die;
  close(WAV_TEST) or die;
  close(TRIAL_OUT) or die;
  close(TRIAL_IN) or die;
}

if (system(
  "utils/utt2spk_to_spk2utt.pl $out_dir/utt2spk >$out_dir/spk2utt") != 0) {
  die "Error creating spk2utt file in directory $out_dir";
}
system("env LC_COLLATE=C utils/fix_data_dir.sh $out_dir");
if (system("env LC_COLLATE=C utils/validate_data_dir.sh --no-text --no-feats $out_dir") != 0) {
  die "Error validating directory $out_dir";
}
