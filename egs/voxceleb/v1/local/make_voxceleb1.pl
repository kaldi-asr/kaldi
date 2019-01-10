#!/usr/bin/perl
#
# Copied from make_voxceleb2.pl and made necessary changes
#
# Usage: make_voxceleb1_v1.pl /export/voxceleb1 dev data/dev
#
# Note: This script requires ffmpeg to be installed and its location included in $PATH.

if (@ARGV != 3) {
  print STDERR "Usage: $0 <path-to-voxceleb1> <dataset> <path-to-data-dir>\n";
  print STDERR "e.g. $0 /export/voxceleb1 dev data/dev\n";
  exit(1);
}

# Check that ffmpeg is installed.
#if (`which ffmpeg` eq "") {
# die "Error: this script requires that ffmpeg is installed.";
#}

($data_base, $dataset, $out_dir) = @ARGV;

if ("$dataset" ne "dev" && "$dataset" ne "test") {
  die "dataset parameter must be 'dev' or 'test'!";
}

opendir my $dh, "$data_base/$dataset/wav" or die "Cannot open directory: $!";
my @spkr_dirs = grep {-d "$data_base/$dataset/wav/$_" && ! /^\.{1,2}$/} readdir($dh);
closedir $dh;

if (system("mkdir -p $out_dir") != 0) {
  die "Error making directory $out_dir";
}

open(SPKR, ">", "$out_dir/utt2spk") or die "Could not open the output file $out_dir/utt2spk";
open(WAV, ">", "$out_dir/wav.scp") or die "Could not open the output file $out_dir/wav.scp";

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
      print WAV "$utt_id", " $wav", "\n";
      print SPKR "$utt_id", " $spkr_id", "\n";
    }
  }
}
close(SPKR) or die;
close(WAV) or die;

if (system(
  "utils/utt2spk_to_spk2utt.pl $out_dir/utt2spk >$out_dir/spk2utt") != 0) {
  die "Error creating spk2utt file in directory $out_dir";
}
system("env LC_COLLATE=C utils/fix_data_dir.sh $out_dir");
if (system("env LC_COLLATE=C utils/validate_data_dir.sh --no-text --no-feats $out_dir") != 0) {
  die "Error validating directory $out_dir";
}

if ("$dataset" eq "test"){
	      my $out_test_dir = "$out_dir";

		if (! -e "$data_base/voxceleb1_test.txt") {
			  system("wget -O $data_base/voxceleb1_test.txt http://www.openslr.org/resources/49/voxceleb1_test.txt");
		  }

		  if (! -e "$data_base/vox1_meta.csv") {
			    system("wget -O $data_base/vox1_meta.csv http://www.openslr.org/resources/49/vox1_meta.csv");
		    }

		    open(TRIAL_IN, "<", "$data_base/voxceleb1_test.txt") or die "Could not open the verification trials file $data_base/voxceleb1_test.txt";
		    open(META_IN, "<", "$data_base/vox1_meta.csv") or die "Could not open the meta data file $data_base/vox1_meta.csv";
		    open(TRIAL_OUT, ">", "$out_test_dir/trials") or die "Could not open the output file $out_test_dir/trials";

		    my %id2spkr = ();
		    my %spkr2id = ();
		    while (<META_IN>) {
			      chomp;
			        my ($vox_id, $spkr_id, $gender, $nation, $set) = split(',');
				  $id2spkr{$vox_id} = $spkr_id;
				    $spkr2id{$spkr_id} = $vox_id;
				    # print "$vox_id \n";
			      }

	            my $test_spkrs = ();
		    while (<TRIAL_IN>) {
			      chomp;
			        my ($tar_or_non, $path1, $path2) = split;
                                   
                                # Create entry for left-hand side of trial
			       my ($spkr_id, $filename) = split('/', $path1);
			       $new_spkr_id = $spkr2id{$spkr_id};
			       my $spkr_id = $new_spkr_id;
		       	       my $rec_id = substr($filename, 0, 11);
			       my $segment = substr($filename, 14, 5);
        		       my $utt_id1 = "$spkr_id-$rec_id-$segment";
                               
                                # Create entry for right-hand side of trial
			        my ($spkr_id, $filename) = split('/', $path2);
				$new_spkr_id = $spkr2id{$spkr_id};
				my $spkr_id = $new_spkr_id;
				my $rec_id = substr($filename, 0, 11);
				my $segment = substr($filename, 14, 5);
				my $utt_id2 = "$spkr_id-$rec_id-$segment";

		        	my $target = "nontarget";
					 if ($tar_or_non eq "1") {
						$target = "target";
							    }
			        print TRIAL_OUT "$utt_id1 $utt_id2 $target\n";
			      }
			      close(TRIAL_OUT) or die;
			      close(TRIAL_IN) or die;
			      close(META_IN) or die;
}
