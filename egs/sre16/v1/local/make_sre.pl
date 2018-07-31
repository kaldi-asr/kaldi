#!/usr/bin/perl
use warnings; #sed replacement for -w perl parameter
#
# Copyright 2015   David Snyder
# Apache 2.0.
# Usage: make_sre.pl <path-to-data> <name-of-source> <sre-ref> <output-dir>

if (@ARGV != 4) {
  print STDERR "Usage: $0 <path-to-data> <name-of-source> <sre-ref> <output-dir>\n";
  print STDERR "e.g. $0 /export/corpora5/LDC/LDC2006S44 sre2004 sre_ref data/sre2004\n";
  exit(1);
}

($db_base, $sre_year, $sre_ref_filename, $out_dir) = @ARGV;
%utt2sph = ();
%spk2gender = ();

$tmp_dir = "$out_dir/tmp";
if (system("mkdir -p $tmp_dir") != 0) {
  die "Error making directory $tmp_dir";
}

if (system("find -L $db_base -name '*.sph' > $tmp_dir/sph.list") != 0) {
  die "Error getting list of sph files";
}
open(WAVLIST, "<$tmp_dir/sph.list") or die "cannot open wav list";

while(<WAVLIST>) {
  chomp;
  $sph = $_;
  @A1 = split("/",$sph);
  @A2 = split("[./]",$A1[$#A1]);
  $uttId=$A2[0];
  $utt2sph{$uttId} = $sph;
}

open(GNDR,">", "$out_dir/spk2gender") or die "Could not open the output file $out_dir/spk2gender";
open(SPKR,">", "$out_dir/utt2spk") or die "Could not open the output file $out_dir/utt2spk";
open(WAV,">", "$out_dir/wav.scp") or die "Could not open the output file $out_dir/wav.scp";
open(SRE_REF, "<$sre_ref_filename") or die "Cannot open SRE reference.";
while (<SRE_REF>) {
  chomp;
  ($speaker, $gender, $other_sre_year, $utt_id, $channel) = split(" ", $_);
  $channel_num = "1";
  if ($channel eq "A") {
    $channel_num = "1";
  } else {
    $channel_num = "2";
  }
  $channel = lc $channel;
  if (($other_sre_year eq "sre20$sre_year") and (exists $utt2sph{$utt_id})) {
    $full_utt_id = "$speaker-sre$sre_year-$utt_id-$channel";
    $spk2gender{"$speaker"} = $gender;
    print WAV "$full_utt_id"," sph2pipe -f wav -p -c $channel_num $utt2sph{$utt_id} |\n";
    print SPKR "$full_utt_id $speaker","\n";
  }
}
foreach $speaker (keys %spk2gender) {
  print GNDR "$speaker $spk2gender{$speaker}\n";
}

close(GNDR) || die;
close(SPKR) || die;
close(WAV) || die;
close(SRE_REF) || die;

if (system(
  "utils/utt2spk_to_spk2utt.pl $out_dir/utt2spk >$out_dir/spk2utt") != 0) {
  die "Error creating spk2utt file in directory $out_dir";
}

system("utils/fix_data_dir.sh $out_dir");
if (system("utils/validate_data_dir.sh --no-text --no-feats $out_dir") != 0) {
  die "Error validating directory $out_dir";
}
