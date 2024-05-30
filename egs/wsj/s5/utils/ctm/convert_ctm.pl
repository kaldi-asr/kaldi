#!/usr/bin/env perl

# Copyright 2012  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.

# This takes as standard input a ctm file that's "relative to the utterance",
# i.e. times are measured relative to the beginning of the segments, and it
# uses a "segments" file (format:
# utterance-id recording-id start-time end-time
# ) and a "reco2file_and_channel" file (format:
# recording-id basename-of-file

$skip_unknown=undef;
if ( $ARGV[0] eq "--skip-unknown" ) {
  $skip_unknown=1;
  shift @ARGV;
}

if (@ARGV < 2 || @ARGV > 3) {
  print STDERR "Usage: convert_ctm.pl <segments-file> <reco2file_and_channel-file> [<utterance-ctm>] > real-ctm\n";
  exit(1);
}

$segments = shift @ARGV;
$reco2file_and_channel = shift @ARGV;

open(S, "<$segments") || die "opening segments file $segments";
while(<S>) {
  @A = split(" ", $_);
  @A == 4 || die "Bad line in segments file: $_";
  ($utt, $recording_id, $begin_time, $end_time) = @A;
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


# Now process the ctm file, which is either the standard input or the third
# command-line argument.
$num_done = 0;
while(<>) {
  @A= split(" ", $_);
  ( @A == 5 || @A == 6 ) || die "Unexpected ctm format: $_";
  # lines look like:
  # <utterance-id> 1 <begin-time> <length> <word> [ confidence ]
  ($utt, $one, $wbegin, $wlen, $w, $conf) = @A;
  $reco = $utt2reco{$utt};
  if (!defined $reco) { 
      next if defined $skip_unknown;
      die "Utterance-id $utt not defined in segments file $segments"; 
  }
  $file = $reco2file{$reco};
  $channel = $reco2channel{$reco};
  if (!defined $file || !defined $channel) { 
    die "Recording-id $reco not defined in reco2file_and_channel file $reco2file_and_channel"; 
  }
  $b = $begin{$utt};
  $e = $end{$utt};
  $wbegin_r = $wbegin + $b; # Make it relative to beginning of the recording.
  $wbegin_r = sprintf("%.2f", $wbegin_r);
  $wlen = sprintf("%.2f", $wlen);
  if (defined $conf) {
    $line = "$file $channel $wbegin_r $wlen $w $conf\n"; 
  } else {
    $line = "$file $channel $wbegin_r $wlen $w\n"; 
  }
  if ($wbegin_r + $wlen > $e + 0.01) {
    print STDERR "Warning: word appears to be past end of recording; line is $line";
  }
  print $line; # goes to stdout.
  $num_done++;
}

if ($num_done == 0) { exit 1; } else { exit 0; }

__END__

# Test example [also test it without the 0.5's]
echo utt reco 10.0 20.0 > segments
echo reco file A > reco2file_and_channel
echo utt 1 8.0 1.0 word 0.5 > ctm_in
echo file A 18.00 1.00 word 0.5 > ctm_out
utils/convert_ctm.pl segments reco2file_and_channel ctm_in | cmp - ctm_out || echo error
rm segments reco2file_and_channel ctm_in ctm_out




