#!/usr/bin/perl

use File::Basename;

# Copyright 2013   Daniel Povey
# Apache 2.0.


if (@ARGV != 3 ) {
  print STDERR "Usage: make_fisher.pl <tbl-file> <sph-list> <out-dir>\n" .
    "e.g.: make_fisher.pl /mnt/data/LDC2004T19/fe_03_p1_tran/doc/fe_03_p1_calldata.tbl " .
    "all_files.txt data/train_fisher\n";
}

($tbl_file, $sph_list_file ,$out_dir) = @ARGV;


open(TBL, "<$tbl_file")  or die "cannot open $tbl_file";
open(SPHLIST, "<", $sph_list_file) or die "cannot open wav list $sph_list_file";

open(GNDR, ">$out_dir/spk2gender") or die "Could not open the output file $out_dir/spk2gender";
open(UTT2SPK, ">$out_dir/utt2spk") or die "Could not open the output file $out_dir/utt2spk";
open(WAV, ">$out_dir/wav.scp") or die "Could not open the output file $out_dir/wav.scp";

@bad_audio = ("01243", "06716", "00446");

# Read the sph-list, this will give us the full pathnames and will help us
# exclude missing files.
while(<SPHLIST>) {
    chop; # e.g. $_ = /export/corpora3/LDC/LDC2004S13/fe_03_p1_sph1/audio/000/fe_03_00037.sph
    $basename = basename($_); # e.g. basename = fe_03_00037.sph
    $basename =~ m/^fe_\d\d_(\d\d\d\d\d)\.sph$/ || die "Unexpected filename $_";
    $utt_id = $1; # match the 5-digit sequence..
    if (/$utt_id/i ~~ @bad_audio) {
      # don't do anything
    } else {
      $wav{$utt_id} = $_;
    }
}

$header = <TBL>; # read the first line
# which is:
# CALL_ID,DATE_TIME,TOPICID,SIG_GRADE,CNV_GRADE,APIN,ASX.DL,APHNUM,APHSET,APHTYP,BPIN,BSX.DL,BPHNUM,BPHSET,BPHTYP
# Note: the numeric pin "APIN" is supposed to correspond to the speaker identitity but it
# does not always, as some individuals shared the pin with others; as a result, some
# PINs are recorded as male and female in separate calls.  To get around this, we just
# regard each PIN as having two versions, a "male" and "female" version, so a PIN 12345
# will be mapped to two PINs, 12345f and 12345m, one or both of which may actually appear.

$num_bad_files = 0;
$num_good_files = 0;

while(<TBL>){
  @conv = split(",",$_);
  @conv == 15 || die "Bad line $_";
  $utt_id = $conv[0];
  $genderA = substr($conv[6], 0, 1); # "m" or "f"
  $spkidA = $conv[5] . $genderA;  # $conv[5] is a numeric PIN (APIN)
  $genderB = substr($conv[11], 0, 1); # "m" or "f"
  $spkidB = $conv[10] . $genderB;

  if (!defined $wav{$utt_id}) {
    $num_bad_files++;
	# print STDERR "no wav file for $utt_id\n";
  } else {
    # we prepend the speaker-id to the utterance-id; this helps ensure
    # that if we sort by utterance, the resulting list has the utterances
    # from a single speaker as a block.
	print WAV "$spkidA-$utt_id", "_A sph2pipe -f wav -p -c 1 $wav{$utt_id} |\n";
	print WAV "$spkidB-$utt_id", "_B sph2pipe -f wav -p -c 2 $wav{$utt_id} |\n";
	print UTT2SPK "$spkidA-$utt_id", "_A $spkidA\n";
	print UTT2SPK "$spkidB-$utt_id", "_B $spkidB\n";

    if (!defined $seen_spk{$spkidA}) {
      $seen_spk{$spkidA} = 1;
      print GNDR "$spkidA $genderA\n";
    }
    if (!defined $seen_spk{$spkidB}) {
      $seen_spk{$spkidB} = 1;
      print GNDR "$spkidB $genderB\n";
    }
	$used{$utt_id} = 1;
    $num_good_files++;
  }
}
while (($key, $value) = each(%wav)) {
  if (!$used{$key}) {
	print STDERR "wav file $value had no corresponding demographic\n";
  }
}

print STDERR "Processed $num_good_files utterances; $num_bad_files had missing wav data.\n";
