#!/usr/bin/env perl
use Getopt::Long;
use Encode;

################################################################################
#
# Script to prepare a NIST .stm file for scoring ASR output.  Based on the files
# that are naturally created for Kaldi acoustic training:
#
#  -  data/segments: contains segmentID, recordingID, start-time & end-time
#
#  -  data/wav.scp: contains recordingID & waveform-name (or sph2pipe command)
#
#  -  data/utt2spk: contains segmentID % speakerID
#
#  -  data/text: contains segment ID and transcription
#
# The .stm file has lines of the form
#
#    waveform-name channel speakerID start-time end-time [<attr>] transcription
#
# Clearly, most of the information needed for creating the STM file is present
# in the four Kaldi files mentioned above, except channel --- its value will be
# obtained from the sph2pipe command if present, or will default to "1" --- and
# <attributes> from a separate demographics.tsv file. (A feature to add later?)
#
# Note: Some text filtering is done by this script, such as removing non-speech
#       tokens from the transcription, e.g. <cough>, <breath>, etc.

        $fragMarkers = ""; # If given by the user, they are stripped from words

#       But two types of tokens are retained as is, if present.
#
        $Hesitation = "<hes>"; # which captures hesitations, filled pauses, etc.
        $OOV_symbol = "<unk>"; # which our system outputs occasionally.
#
# Note: The .stm file must be sorted by filename and channel in ASCII order and
#       by the start=time in numerical order.  NIST recommends the unix command
#       "sort +0 -1 +1 -2 +3nb -4"
#
# This script will also produce an auxilliary file named reco2file_and_channel
# which is used by Kaldi scripts to produce output in .ctm format for scoring.
# So any channel ID assigned here will be consistent between ref and output.
#
# If the training text is Viterbi-aligned to the speech to obtain time marks,
# it should be straightforward to modify this script to produce a .ctm file:
#
#    waveform-file channel start-time duration word
#
# which lists the transcriptions with word-level time marks.
#
# Note: A .ctm file must be sorted via "sort +0 -1 +1 -2 +2nb -3"
#
################################################################################
GetOptions("fragmentMarkers=s" => \$fragMarkers, "hesitationToken=s" => \$Hesitation,"oovToken=s" => \$OOV_symbol);

if ($#ARGV == 0) {
    $inDir = $ARGV[0];
    print STDERR ("$0: Making stm file from information in $inDir\n");
    print STDERR ("\tRemoving [$fragMarkers]+ from ends of tokens\n") if ($fragMarkers);
    print STDERR ("\tPreserving hesitation tokens $Hesitation\n") unless ($Hesitation eq "<hes>");
    print STDERR ("\tUsing $OOV_symbol as the OOV symbol\n") unless ($OOV_symbol eq "<unk>");
} else {
    print STDERR ("Usage: $0 [--options] DataDir\n");
    print STDERR ("\t--fragmentMarkers <chars>  Strip these from ends of each token (default: none)\n");
    print STDERR ("\t--hesitationToken <foo>    Preserve <foo> when deleting non-speech tokens (default: <hes>)\n");
    print STDERR ("\t--oovToken <bar>           Use <bar> to replace hard-coded OOVs (default: <unk>)\n");
    exit(1);
}

$segmentsFile = "$inDir/segments";
$scpFile      = "$inDir/wav.scp";
$utt2spkFile  = "$inDir/utt2spk";
$textFile     = "$inDir/text";
$stmFile      = "$inDir/stm";
$charStmFile  = "$inDir/char.stm";
$reco2ctmFile = "$inDir/reco2file_and_channel";

################################################################################
# Read the segmentIDs, file-IDs, start- and end-times from the segments file
################################################################################

die "Current version of script requires a segments file" unless (-e $segmentsFile);

open(SEGMENTS, $segmentsFile)
    || die "Unable to read segments file $segmentsFile";
$numSegments = 0;
while ($line=<SEGMENTS>) {
  @tokens = split(/\s+/, $line);
  unless ($#tokens == 3) {
    print STDERR ("$0: Couldn't parse line $. in $segmentsFile\n\t$line\n");
    next;
  }
  $segmentID = shift @tokens;
  if (exists $fileID{$segmentID}) {
    print STDERR ("$0: Skipping duplicate segment ID $segmentID in $segmentsFile\n");
    next;
  }
  $fileID{$segmentID}    = shift @tokens;
  $startTime{$segmentID} = shift @tokens;
  $endTime{$segmentID}   = shift @tokens;
  ++$numSegments;
}
close(SEGMENTS);
print STDERR ("$0: Read info about $numSegments segment IDs from $segmentsFile\n");

################################################################################
# Read the waveform filenames from the wav.scp file.  (Parse sph2pipe command.)
################################################################################

open(SCP, $scpFile)
    || die "Unable to open scp file $scpFile\n";
$numRecordings = 0;
while ($line=<SCP>) {
    chomp;
    if ($line =~ m:^\s*(\S+)\s+(.+)$:) {
        $recordingID  = $1;
        $waveformFile = $2;
    } else {
      print STDERR ("$0: Couldn't parse line $. in $scpFile\n\t$line\n");
      next;
    }
    if (exists $waveform{$recordingID}) {
        print STDERR ("$0: Skipping duplicate recording ID $recordingID in $scpFile\n");
        # BUG ALERT: This check may need to be turned off for multi-channel recordings,
        #            since the same recording may appear with with different channels?
        next;
    }
    if ($waveformFile =~ m:^\S+$:) {
      # This is a single filename, no shp2pipe or gunzip for reading waveforms
      $waveform{$recordingID} = $waveformFile;
    } elsif (($waveformFile =~ m:(sph2pipe|gunzip|gzip|cat|zcat)\s+:) &&
             ($waveformFile =~ m:\s+(\S+)\s*\|$:)) {
      # HACK ALERT: the filename is *assumed* to be at the END of the command
      $waveform{$recordingID} = $1;
      $channel{$recordingID}  = $1 if ($waveformFile =~ m:sph2pipe\s+.*\-c\s+(\S+)\s+.+:);
    } elsif (($waveformFile =~ m:(sox)\s+:) &&
             ($waveformFile =~ m:\s+(\S+)\s*\|$:)) {
      # HACK ALERT: the first element that does ends with '.wav' is assumed to
      # be the original filename
      @elems=split(/\s+/, $waveformFile);
      foreach $elem (@elems) {
        if ($elem =~ m/.*\.wav/) {
          $filename=$elem;
          last;
        }
      }
      die ("$0: Couldn't parse waveform filename on line $. in $scpFile\n\t$line\n") if not defined $filename;
      die ("$0: Filename $filename does not exist: in $scpFile\n\t$line\n") unless (-e $filename);

      $waveform{$recordingID} = $filename;
      #$channel{$recordingID}  = $filename;
    } else {
      print STDERR ("$0: Couldn't parse waveform filename on line $. in $scpFile\n\t$line\n");
      next;
    }
    $waveform{$recordingID} =~ s:.+/::;             # remove path prefix
    $waveform{$recordingID} =~ s:\.(sph|wav)\s*$::; # remove file extension
    $channel{$recordingID} = 1                      # Default
      unless (exists $channel{$recordingID});
    ++$numRecordings;
}
close(SCP);
print STDERR ("$0: Read filenames for $numRecordings recording IDs from $scpFile\n");

################################################################################
# Read speaker information from the utt2spk file
################################################################################

open(UTT2SPK, $utt2spkFile)
    || die "Unable to read utt2spk file $utt2spkFile";
$numSegments = 0;
while ($line=<UTT2SPK>) {
    @tokens = split(/\s+/, $line);
    if (! ($#tokens == 1)) {
        print STDERR ("$0: Couldn't parse line $. in $utt2spkFile\n\t$line\n");
        next;
    }
    $segmentID = shift @tokens;
    if (exists $speakerID{$segmentID}) {
        print STDERR ("$0: Skipping duplicate segment ID $segmentID in $utt2spkFile\n");
        next;
    }
    $speakerID{$segmentID} = shift @tokens;
    ++$numSegments;
}
close(UTT2SPK);
print STDERR ("$0: Read speaker IDs for $numSegments segments from $utt2spkFile\n");

################################################################################
# Read the transcriptions from the text file
################################################################################

open(TEXT, $textFile)
    || die "Unable to read text file $textFile";
$numSegments = $numWords = 0;
while ($line=<TEXT>) {
    chomp;
    if ($line =~ m:^(\S+)\s+(.+)$:) {
        $segmentID   = $1;
        $text          = $2;
    } else {
      print STDERR ("$0: Couldn't parse line $. in $textFile\n\t$line\n");
      next;
    }
    if (exists $transcription{$segmentID}) {
        print STDERR ("$0: Skipping duplicate segment ID $segmentID in $segmentsFile\n");
        next;
    }
    $transcription{$segmentID} = "";
    @tokens = split(/\s+/, $text);
    # This is where one could filter the transcription as necessary.
    # E.g. remove noise tokens, mark non-scoring segments, etc.
    # HACK ALERT: Current version does this is an ad hoc manner!
    while ($w = shift(@tokens)) {
        # Substitute OOV tokens specific to the Babel data
        $w = $OOV_symbol if ($w eq "(())");
        # Remove fragMarkers, if provided, from either end of the word
        $w =~ s:(^[$fragMarkers]|[$fragMarkers]$)::g if ($fragMarkers);
        # Omit non-speech symbols such as <cough>, <breath>, etc.
        $w =~ s:^<[^>]+>$:: unless (($w eq $OOV_symbol) || ($w eq $Hesitation));
        next if ($w eq "");
        $transcription{$segmentID} .= " $w";
        $numWords++;
    }
    $transcription{$segmentID} =~ s:^\s+::;  # Remove leading white space
    $transcription{$segmentID} =~ s:\s+$::;  # Remove training white space
    $transcription{$segmentID} =~ s:\s+: :g; # Normalize remaining white space
    # Transcriptions containing no words, or only OOVs and hesitations are not scored
    $transcription{$segmentID} = "IGNORE_TIME_SEGMENT_IN_SCORING"
        if (($transcription{$segmentID} eq "") ||
            ($transcription{$segmentID} =~ m:^(($OOV_symbol|$Hesitation)\s*)+$:));
    ++$numSegments;
}
close(TEXT);
print STDERR ("$0: Read transcriptions for $numSegments segments ($numWords words) from $textFile\n");

################################################################################
# Write the transcriptions in stm format to a file named stm
################################################################################

print STDERR ("$0: Overwriting existing stm file $stmFile\n")
    if (-s $stmFile);
open(STM, "| sort +0 -1 +1 -2 +3nb -4 > $stmFile")
    || die "Unable to write to stm file $stmFile";
$numSegments = 0;
foreach $segmentID (sort keys %fileID) {
    if (exists $waveform{$fileID{$segmentID}}) {
        printf STM ("%s %s %s %.2f %.2f",
                    $waveform{$fileID{$segmentID}},
                    $channel{$fileID{$segmentID}},
                    $speakerID{$segmentID},
                    $startTime{$segmentID},
                    $endTime{$segmentID});
        printf STM (" <%s>", $attributes{$segmentID}) if (exists $attributes{$segmentID});
        printf STM (" %s\n", $transcription{$segmentID});
        ++$numSegments;
    } else {
        print STDERR ("$0: No waveform found for segment $segmentID, file $fileID{$segmentID}\n");
    }
}
close(STM);
print STDERR ("$0: Wrote reference transcriptions for $numSegments segments to $stmFile\n");

################################################################################
# Write a character-separated stm file as well, for CER computation
################################################################################

print STDERR ("$0: Overwriting existing stm file $charStmFile\n")
    if (-s $charStmFile);
open(STM, "$stmFile")
    || die "Unable to read back stm file $stmFile";
binmode STM,":encoding(utf8)";
open(CHARSTM, "> $charStmFile")
    || die "Unable to write to char.stm file $charStmFile";
binmode CHARSTM,":encoding(utf8)";
while ($line=<STM>) {
    @tokens = split(/\s+/, $line);
    # The first 5 tokens are filename, channel, speaker, start- and end-time
    for ($n=0; $n<5; $n++) {
        $w = shift @tokens;
        print CHARSTM ("$w ");
    }
    # CER is used only for some scripts, e.g. CJK.  So only non-ASCII characters
    # in the remaining tokens should be split into individual tokens.
    $w = join (" ", @tokens);
    $w =~ s:([^\x00-\x7F])(?=[^\x00-\x7F]):$1 :g; # split adjacent non-ASCII chars
    print CHARSTM ("$w\n");
}
close(CHARSTM);
close(STM);
print STDERR ("$0: Wrote char.stm file $charStmFile\n");

################################################################################
# Write the reco2file_and_channel file for use by Kaldi scripts
################################################################################

print STDERR ("$0: Overwriting existing reco2file_and_channel file $reco2ctmFile\n")
    if (-s $reco2ctmFile);
open(RECO2CTM, "| sort > $reco2ctmFile")
    || die "Unable to write to reco2file_and_channel file $reco2ctmFile";
$numRecordings = 0;
foreach $recordingID (sort keys %waveform) {
    printf RECO2CTM ("%s %s %s\n", $recordingID, $waveform{$recordingID}, $channel{$recordingID});
    ++$numRecordings;
}
close(RECO2CTM);
print STDERR ("$0: Wrote file_and_channel info for $numRecordings recordings to $reco2ctmFile\n");

print STDERR ("$0: Done!\n");
exit(0);
