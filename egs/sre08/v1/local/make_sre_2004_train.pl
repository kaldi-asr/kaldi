#!/usr/bin/env perl
#
# Copyright 2013   Daniel Povey
# Apache 2.0.
# Usage: make_sre_2004_train.pl <path to LDC2006S44> <Path to root level output dir>

if (@ARGV != 2) {
  print STDERR "Usage: $0 <path-to-LDC2006S44> <path-to-output>\n";
  print STDERR "e.g. $0 /export/corpora5/LDC/LDC2006S44 data\n";
  exit(1);
}

($db_base, $out_base_dir) = @ARGV;

$tmp_dir = "$out_base_dir/tmp";
if (system("mkdir -p $tmp_dir") != 0) {
  die "Error making directory $tmp_dir";
}

if (system("find $db_base -name '*.sph' > $tmp_dir/sph.list") != 0) {
  die "Error getting list of sph files";
}
open(WAVLIST, "<", "$tmp_dir/sph.list") or die "cannot open wav list";

while(<WAVLIST>) {
  chomp;
  $sph = $_;
  @t = split("/",$sph);
  @t1 = split("[./]",$t[$#t]);
  $uttId=$t1[0];
  $wav{$uttId} = $sph;
}

@gender_list=("male","female");
foreach (@gender_list) {
  $gender=$_;
  $g = substr($gender, 0, 1);
  @case_list=("10sec","16sides","1side","30sec","3convs","3sides","8sides");
  foreach $case (@case_list) {
    $out_dir = "$out_base_dir/sre04_train_${case}_${gender}";
    mkdir "$out_dir";
    $casefile = $db_base."/r93_6_1/sp04-06/train/".$gender."/".$case.".trn";
    open(CF, "<", $casefile)  or die "cannot open $casefile";
    open(GNDR,">", "$out_dir/spk2gender") or die "Could not open the output file $out_dir/spk2gender";
    open(SPKR,">", "$out_dir/utt2spk") or die "Could not open the output file $out_dir/utt2spk";
    open(WAV,">", "$out_dir/wav.scp") or die "Could not open the output file $out_dir/wav.scp";
    while (<CF>) {
      chomp;
      $line = $_;
      @A = split(" ",$line);
      $spkr = $A[0];
      print GNDR "$spkr $g\n";
      @wav_list = split(",", $A[1]);
      foreach $wav_id (@wav_list) {
        @B = split("[.]",$wav_id);
        $uttId = $B[0];
        $wav = $wav{$uttId};
        $uttId = $spkr . "-" . $uttId;
        if ($wav && -e $wav) {
          print WAV "$uttId"," sph2pipe -f wav -p $wav |\n";
          print SPKR "$uttId"," $spkr","\n";
        } else {
          print STDERR "Missing $wav\n";
        }
      }
    }
    close(GNDR) || die;
    close(SPKR) || die;
    close(WAV) || die;
    if (system("utils/utt2spk_to_spk2utt.pl $out_dir/utt2spk >$out_dir/spk2utt") != 0) {
      die "Error creating spk2utt file in directory $out_dir";
    }
    system("utils/fix_data_dir.sh $out_dir");
    if (system("utils/validate_data_dir.sh --no-text --no-feats $out_dir") != 0) {
      die "Error validating directory $out_dir";
    }
  }
}
