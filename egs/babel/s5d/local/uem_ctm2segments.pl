#!/usr/bin/env perl
use Getopt::Long;

################################################################################
# Convert a CTM file produced by decoding a long segment, typically several min
# long, into a sequence of shorter segments of duration 10-15 seconds.  Produce
# a segments file of the form used for Kaldi training/decoding
#
#       utteranceID recordingID startTime endTime
#
# The desired outcome is that the long (input) segment will be recursively cut
# into shorter segments at the location of long silences, leaving (say) 0.5 sec
# of silence at each end of the two resulting shorter segments, until all the
# segments are of the desired duration.
#
# NOTE: It is assumed that the CTM file provides time information at 0.01 sec
#       resolution, and that any missing segments in the CTM correspond to the
#       optional silence model, whose output token was removed by the sequence
#
#       lattice-align-words --> lattice-to-ctm-conf --> raw CTM file
#
        $ctmTimeStep = 0.01; # Could be changed if needed by --timeStep
#
#       It is further assumed that the explicit silence token (word) is
#
        $silence = "<silence>";
#
#       This could be changed using the --silence option if needed.
#
#       Another option is the minimum silence duration to permit segmentation
#
        $minSilence = 1.02; # seconds
#
#       Maximum allowed segment length, could be changed through --maxSegLen
#
        $maxSegLen = 30; #seconds
#
#       Default segment length, used when the ctm segment is too long
#
        $defaultSegLen = 10; # seconds
################################################################################

GetOptions("ctmTimeStep=f" => \$ctmTimeStep, 
  "minSilence=f" => \$minSilence, 
  "silence=s" => \$silence, 
  "maxSegLen=f" => \$maxSegLen, 
  "defaultSegLen=f" => \$defaultSegLen);

if ($#ARGV == 1) {
  $ctmFile  = $ARGV[0];
  $segmentsFile = $ARGV[1];
  print STDERR ("$0: $ctmFile $segmentsFile\n");
  print STDERR ("\t--ctmTimeStep = $ctmTimeStep\n") unless ($ctmTimeStep == 0.01);
  print STDERR ("\t--silence = $silence\n") unless ($silence eq "<silence>");
  print STDERR ("\t--maxSegLen = $maxSegLen\n") unless ($maxSegLen == 30);
  print STDERR ("\t--defaultSegLen = $defaultSegLen\n") unless ($defaultSegLen == 10);

} else {
  print STDERR ("Usage: $0 [--options] inputCTM outputSegments\n");
  print STDERR ("\t--ctmTimeStep   %f   Time resolution of CTM file (default 0.01 sec)\n");
  print STDERR ("\t--silence       %s   Word token for silence (default <silence>)\n");
  print STDERR ("\t--maxSegLen     %f   Max allowed segment length (default 30 sec)\n");
  print STDERR ("\t--defaultSegLen %f   Default segment length (default 10 sec)\n");
  exit(1);
}

open (CTM, $ctmFile)
|| die "Unable to open input CTM file $ctmFile for reading";
$numRecordings  = $numWords = $n = 0;
$prevFileName   = "";
$prevChannel    = "";
$prevEndTime    = 0.00;
$prevConfidence = 0.00;
while ($line=<CTM>) {
  @token = split(/\s+/, $line);
  unless (($#token==4)||($#token==5)) {
    # CTM should have 5 or 6 tokens per line
    # audioFile channel startTime duration word [confidence]
    print STDERR ("$0 WARNING: unparsable line $. in ctm file: $line");
    next;
  }
  if ( ( ($token[0] ne $prevFileName) || ($token[1] ne $prevChannel) ) && ($prevFileName ne "") ) {
    break if ($n==0);
    ########################################################################
    # This is the next audio file; create segments for the previous file
    ########################################################################
    print STDERR ("Audio file $prevFileName contains $n word tokens\n");
    printf STDERR ("\t%d alternating speech/silence segments after mergers\n", &process_this_audio_file);
    ########################################################################
    # Done writing out the segments for the previous audio recording
    ########################################################################
    $numRecordings++;
    # Reset to process the next file
    $prevFileName   = "";
    $prevChannel    = "";
    $prevEndTime    = 0.00;
    $prevConfidence = 0.00;
    $n=0;
  }
  # Otherwise, this is the next word in the same (i.e. previous) audio file
  if ( ($token[2]-$prevEndTime) > $ctmTimeStep ) {
    # There is a missing segment in the CTM, presumably silence
    $fileName[$n]  = $token[0];
    $channel[$n]   = $token[1];
    $startTime[$n] = $prevEndTime;
    $endTime[$n]   = $token[2];
    $wordToken[$n] = $silence;
    $confidence[$n]= $prevConfidence;
    $n++;
  }
  # Record this token for processing later
  $prevFileName   = $fileName[$n]   = $token[0];
  $prevChannel    = $channel[$n]    = $token[1];
  $startTime[$n]  = $token[2];
  $prevEndTime    = $endTime[$n]    = ($token[2]+$token[3]);
  $wordToken[$n]  = $token[4];
  $prevConfidence = $confidence[$n] = $token[5] if ($#token==5);
  $n++;
  $numWords++;
}
close(CTM);
if ($n>0) {
  # This is the last audio file; create segments for the file
  print STDERR ("Audio file $prevFileName contains $n word tokens\n");
  printf STDERR ("\t%d alternating speech/silence segments after mergers\n", &process_this_audio_file);
  # Done writing out the segments for the last audio recording
  $numRecordings++;
}
print STDERR ("Read $numRecordings filenames containing $numWords words from $ctmFile\n");


sub process_this_audio_file {
  # Merge consecutive speech/silence tokens to create candidate "segments"
  $s=0;
  $segmentStart[$s] = 0.00;
  $segmentType[$s]  = $silence;
  $segmentEnd[$s]   = -1.0;
  for ($i=0; $i<$n; $i++) {
    $sTime = $startTime[$i];
    $word  = $wordToken[$i];
    $eTime = $endTime[$i];
    if ( ($word eq $silence) && ($segmentType[$s] ne $silence)
      || ($word ne $silence) && ($segmentType[$s] eq $silence) ) {
      $segmentEnd[$s] = $sTime;
      $s++;
      $segmentStart[$s] = $sTime;
      $segmentType[$s]  = ($word eq $silence) ? $silence : "<speech>" ;
    }
    $segmentEnd[$s] = $eTime;
  }
  # Merge speech segments separated by silence of less than some minimum duration
  # Note: there must be at least two segments for mergers to be an option, i.e. $s>0.
  if ($s>0) {
    if ( ($segmentType[0] eq $silence)
      && ( ($segmentEnd[0]-$segmentStart[0]) < $minSilence) ) {
      die "Something wrong: initial silence segment must have a speech segment following it"
      unless ($segmentType[1] eq "<speech>");
      $segmentType[0] = $segmentType[1];
      $segmentEnd[0]  = $segmentEnd[1];
      for ($j=2; $j<=$s; $j++) {
        $segmentStart[$j-1] = $segmentStart[$j];
        $segmentType[$j-1]  = $segmentType[$j];
        $segmentEnd[$j-1]   = $segmentEnd[$j];
      }
      $s--; # one silence segment removed
    }
    for ($i=1; $i<$s; $i++) {
      if ( ($segmentType[$i] eq $silence)
        && ( ($segmentEnd[$i]-$segmentStart[$i]) < $minSilence) ) {
        die "Something wrong: internal silence segment must have speech segments on eithe side"
        unless ( ($segmentType[$i-1] eq "<speech>") && ($segmentType[$i+1] eq "<speech>") );
        $segmentEnd[$i-1] = $segmentEnd[$i+1];
        for ($j=$i+2; $j<=$s; $j++) {
          $segmentStart[$j-2] = $segmentStart[$j];
          $segmentType[$j-2]  = $segmentType[$j];
          $segmentEnd[$j-2]   = $segmentEnd[$j];
        }
        $s -= 2; # one silence removed, two speech segments merged
        $i--;    # backtrack, to process the segment that just moved into position $i
      }
    }
    if ( ($segmentType[$s] eq $silence)
      && ( ($segmentEnd[$s]-$segmentStart[$s]) < $minSilence) ) {
      die "Something wrong: final silence segment must have a speech segment preceding it"
      unless ($segmentType[$s-1] eq "<speech>");
      $segmentEnd[$s-1]  = $segmentEnd[$s];
      $s--; # one silence segment removed
    }
  }
  # Print segment markers for debugging
  $num = $s + 1;
  for ($i=0; $i<=$s; $i++) {
#	printf STDOUT ("%s %s %.2f %.2f %s\n",
#    printf STDOUT ("%s %s %.2f %.2f\n",
#      sprintf ("%s_%06i",$prevFileName,(100*$segmentStart[$i])),
#      $prevFileName,
#		       $segmentStart[$i],
#		       $segmentEnd[$i], $segmentType[$i]);
#      ($segmentStart[$i] - (($i==0) ? 0.0 : 0.5)),
#      ($segmentEnd[$i] + (($i==$s) ? 0.0 : 0.5))) unless ($segmentType[$i] eq $silence);
    if ($segmentType[$i] ne $silence) {
      if (($segmentEnd[$i] - $segmentStart[$i]) > $maxSegLen) {
        $fakeStart = $segmentStart[$i] - (($i==0) ? 0.0 : 0.5);
        while (($segmentEnd[$i] - $fakeStart) > $defaultSegLen) {
          printf STDOUT ("%s %s %.2f %.2f\n",
            sprintf ("%s_%06i",$prevFileName,(100*$fakeStart)),
            $prevFileName,
            $fakeStart,
            $fakeStart + $defaultSegLen);
          $fakeStart += $defaultSegLen;
          $num += 2;
        }
        if (($segmentEnd[$i] - $fakeStart) > 0) {
          printf STDOUT ("%s %s %.2f %.2f\n",
            sprintf ("%s_%06i",$prevFileName,(100*$fakeStart)),
            $prevFileName,
            $fakeStart,
            ($segmentEnd[$i] + (($i==$s) ? 0.0 : 0.5)));
        } else {
          $num -= 2;
        }
      } else {
        printf STDOUT ("%s %s %.2f %.2f\n",
          sprintf ("%s_%06i",$prevFileName,(100*$segmentStart[$i])),
          $prevFileName,
          ($segmentStart[$i] - (($i==0) ? 0.0 : 0.5)),
          ($segmentEnd[$i] + (($i==$s) ? 0.0 : 0.5)));
      }
    }
  }
  $num;
}
