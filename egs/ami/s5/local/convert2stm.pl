#!/usr/bin/perl

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
  @A > 4 || die "Bad line in segments file: $_";
  ($utt, $recording_id, $begin_time, $end_time) = @A[0..3];
  $utt2reco{$utt} = $recording_id;
  $begin{$utt} = $begin_time;
  $end{$utt} = $end_time;
}
close(S);

open(R, "<$reco2file_and_channel") || die "open reco2file_and_channel file $reco2file_and_channel";
while(<R>) {
  @A = split(" ", $_);
  @A == 3 || die "Bad line in reco2file_and_channel file: $_";
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
  @A == 2 || die "Bad line in utt2spk file: $_";
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
    die "Recording-id $recording_id not defined in reco2file_and_channel file $reco2file_and_channel"; 
  }
 
  $speaker = $utt2spk{$utt};
  $transcripts = $utt2text{$utt};  
  
  if (!defined $speaker) { die "Speaker-id for utterance $utt not defined in utt2spk file $utt2spk_file"; }
  if (!defined $transcripts) { die "Transcript for $utt not defined in text file $text"; }

  $b = $begin{$utt};
  $e = $end{$utt};
  $line = "$file $channel $speaker $b $e $transcripts \n";

  print $line; # goes to stdout.
}

__END__

# Test example [also test it without the 0.5's]
echo utt reco 10.0 20.0 > segments
echo reco file A > reco2file_and_channel
echo utt 1 8.0 1.0 word 0.5 > ctm_in
echo file A 18.00 1.00 word 0.5 > ctm_out
utils/convert_ctm.pl segments reco2file_and_channel ctm_in | cmp - ctm_out || echo error
rm segments reco2file_and_channel ctm_in ctm_out




