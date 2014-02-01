#!/usr/bin/perl
#
# Copyright 2013   Daniel Povey
# Apache 2.0.
# Usage: make_sre_2006_train.pl <path to LDC2011S09> <Path to root level output dir>

if (@ARGV != 2) {
  print STDERR "Usage: $0 <path-to-LDC2011S09> <path-to-output>\n";
  print STDERR "e.g. $0 /export/corpora5/LDC/LDC2011S09 data\n";
  exit(1);
}

($db_base, $out_base_dir) = @ARGV;

@gender_list=("male","female");
foreach (@gender_list) {
  $gender=$_;
  $g = substr($gender, 0, 1);
  @case_list=("10sec4w","1conv4w","3conv4w","8conv4w");
  foreach $case (@case_list) {
    $out_dir = "$out_base_dir/sre06_train_${case}_${gender}";
    mkdir "$out_dir";
    $casefile = $db_base."/data/train/".$gender."/".$case.".trn";
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
        @B = split(":",$wav_id);
        $basename = $B[0];
        $wav = $db_base."/data/train/data/$basename";
        $side = $B[1];
        $raw_basename = $basename;
        $raw_basename =~ s/.sph//;
        $uttId = $spkr . "-" . $raw_basename . "_" . $side;
        if ( $side eq "A" ) {
          $channel = 1;
        } elsif ( $side eq "B" ) {
          $channel = 2;
        } else {
          die "unknown channel $side\n";
        }
        if ($wav && -e $wav) {
          print WAV "$uttId"," sph2pipe -f wav -p -c $channel $wav |\n";
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
