#!/usr/bin/env perl

# Copyright 2012  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.
#           2013  University of Edinburgh (Author: Pawel Swietojanski)

# This takes as standard input path to directory containing all the usual 
# data files - segments, text, utt2spk and reco2file_and_channel and creates stm

if (@ARGV < 1 || @ARGV > 2) {
  print STDERR "Usage: convert2stm.pl <data-dir> [<utt2spk_stm>] > stm-file\n";
  exit(1);
}

$dir=shift @ARGV;
$utt2spk_file=shift @ARGV || 'utt2spk';

$segments = "$dir/segments";
$reco2file_and_channel = "$dir/reco2file_and_channel";
$text = "$dir/text";
$utt2spk_file = "$dir/$utt2spk_file";

open(S, "<$segments") || die "opening segments file $segments";
while(<S>) {
  @A = split(" ", $_);
  @A > 3 || die "convert2stm: Bad line in segments file: $_";
  ($utt, $recording_id, $begin_time, $end_time) = @A[0..3];
  $utt2reco{$utt} = $recording_id;
  $begin{$utt} = $begin_time;
  $end{$utt} = $end_time;
}
close(S);

open(R, "<$reco2file_and_channel") || die "open reco2file_and_channel file $reco2file_and_channel";
while(<R>) {
  @A = split(" ", $_);
  @A == 3 || die "convert2stm: Bad line in reco2file_and_channel file: $_";
  ($recording_id, $file, $channel) = @A;
  $reco2file{$recording_id} = $file;
  $reco2channel{$recording_id} = $channel;
}
close(R);

open(T, "<$text") || die "open text file $text";
while(<T>) {
  @A = split(" ", $_);
  $utt = shift @A;  
  $utt2text{$utt} = "@A"; 
}
close(T);

open(U, "<$utt2spk_file") || die "open utt2spk file $utt2spk_file";
while(<U>) {
  @A = split(" ", $_);
  @A == 2 || die "convert2stm: Bad line in utt2spk file: $_";
  ($utt, $spk) = @A;
  $utt2spk{$utt} = $spk;
}
close(U);

# Now generate the stm file
foreach $utt (sort keys(%utt2reco)) {

  # lines look like:
  # <File> <Channel> <Speaker> <BeginTime> <EndTime> [ <LABEL> ] transcript
  $recording_id = $utt2reco{$utt};
  if (!defined $recording_id) { die "Utterance-id $utt not defined in segments file $segments"; }
  $file = $reco2file{$recording_id};
  $channel = $reco2channel{$recording_id};
  if (!defined $file || !defined $channel) { 
    die "convert2stm: Recording-id $recording_id not defined in reco2file_and_channel file $reco2file_and_channel"; 
  }
 
  $speaker = $utt2spk{$utt};
  $transcripts = $utt2text{$utt};  
  
  if (!defined $speaker) { die "convert2stm: Speaker-id for utterance $utt not defined in utt2spk file $utt2spk_file"; }
  if (!defined $transcripts) { die "convert2stm: Transcript for $utt not defined in text file $text"; }

  $b = $begin{$utt};
  $e = $end{$utt};
  $line = "$file $channel $speaker $b $e $transcripts \n";

  print $line; # goes to stdout.
}

__END__

# Test example
# ES2011a.Headset-0 A AMI_ES2011a_H00_FEE041 34.27 37.14 HERE WE GO
mkdir tmpdir
echo utt reco 10.0 20.0 > tmpdir/segments
echo utt word > tmpdir/text
echo reco file A > tmpdir/reco2file_and_channel
echo utt spk > tmpdir/utt2spk
echo file A spk 10.0 20.00 word > stm_tst
utils/convert2stm.pl tmpdir | cmp - stm_tst || echo error
rm -r tmpdir stm_tst




