#!/usr/bin/perl

# Copyright 2014  Johns Hopkins University (Author: Sanjeev Khudanpur), Vimal Manohar 
# Apache 2.0

################################################################################
#
# This script was written to check the goodness of automatic segmentation tools
# It assumes input in the form of two Kaldi segments files, i.e. a file each of
# whose lines contain four space-separated values:
#
#    UtteranceID  FileID  StartTime EndTime
#
# It computes # missed frames, # false positives and # overlapping frames.
#
################################################################################

if ($#ARGV == 1) {
    $ReferenceSegmentation = $ARGV[0];
    $HypothesizedSegmentation = $ARGV[1];
    printf STDERR ("Comparing reference segmentation\n\t%s\nwith proposed segmentation\n\t%s\n",
		   $ReferenceSegmentation,
		   $HypothesizedSegmentation);
} else {
    printf STDERR "This program compares the reference segmenation with the proposted segmentation\n";
    printf STDERR "Usage: $0 reference_segments_filename proposed_segments_filename\n";
    printf STDERR "e.g. $0 data/dev10h/segments data/dev10h.seg/segments\n";
    exit (0);
}

################################################################################
# First read the reference segmentation, and
# store the start- and end-times of all segments in each file.
################################################################################

open (SEGMENTS, "cat $ReferenceSegmentation | sort -k2,2 -k3n,3 -k4n,4 |")
    || die "Unable to open $ReferenceSegmentation";
$numLines = 0;
while ($line=<SEGMENTS>) {
    chomp $line;
    @field = split("[ \t]+", $line);
    unless ($#field == 3) {
  exit (1);
	printf STDERR "Skipping unparseable line in file $ReferenceSegmentation\n\t$line\n";
	next;
    }
    $fileID = $field[1];
    unless (exists $firstSeg{$fileID}) {
	$firstSeg{$fileID} = $numLines;
	$actualSpeech{$fileID} = 0.0;
	$hypothesizedSpeech{$fileID} = 0.0;
	$foundSpeech{$fileID} = 0.0;
	$falseAlarm{$fileID} = 0.0;
	$minStartTime{$fileID} = 0.0;
	$maxEndTime{$fileID} = 0.0;
    }
    $refSegName[$numLines] = $field[0];
    $refSegStart[$numLines] = $field[2];
    $refSegEnd[$numLines] = $field[3];
    $actualSpeech{$fileID} += ($field[3]-$field[2]);
    $minStartTime{$fileID} = $field[2] if ($minStartTime{$fileID}>$field[2]);
    $maxEndTime{$fileID} = $field[3] if ($maxEndTime{$fileID}<$field[3]);
    $lastSeg{$fileID} = $numLines;
    ++$numLines;
}
close(SEGMENTS);
print STDERR "Read $numLines segments from $ReferenceSegmentation\n";

################################################################################
# Process hypothesized segments sequentially, and gather speech/nonspeech stats
################################################################################

open (SEGMENTS, "cat $HypothesizedSegmentation | sort -k2,2 -k1,1 |")
    # Kaldi segments files are sorted by UtteranceID, but we re-sort them here
    # so that all segments of a file are read together, sorted by start-time.
    || die "Unable to open $HypothesizedSegmentation";
$numLines = 0;
$totalHypSpeech = 0.0;
$totalFoundSpeech = 0.0;
$totalFalseAlarm = 0.0;
$numShortSegs = 0;
$numLongSegs = 0;
while ($line=<SEGMENTS>) {
    chomp $line;
    @field = split("[ \t]+", $line);
    unless ($#field == 3) {
  exit (1);
	printf STDERR "Skipping unparseable line in file $HypothesizedSegmentation\n\t$line\n";
	next;
    }
    $fileID = $field[1];
    $segStart = $field[2];
    $segEnd = $field[3];
    if (exists $firstSeg{$fileID}) {
	# This FileID exists in the reference segmentation
	# So gather statistics for this UtteranceID
	$hypothesizedSpeech{$fileID} += ($segEnd-$segStart);
	$totalHypSpeech += ($segEnd-$segStart);
	if (($segStart>=$maxEndTime{$fileID}) || ($segEnd<=$minStartTime{$fileID})) {
	    # This entire segment is a false alarm
	    $falseAlarm{$fileID} += ($segEnd-$segStart);
	    $totalFalseAlarm += ($segEnd-$segStart);
	} else {
	    # This segment may overlap one or more reference segments
	    $p = $firstSeg{$fileID};
	    while ($refSegEnd[$p]<=$segStart) {
		++$p;
	    }
	    # The overlap, if any, begins at the reference segment p
	    $q = $lastSeg{$fileID};
	    while ($refSegStart[$q]>=$segEnd) {
		--$q;
	    }
	    # The overlap, if any, ends at the reference segment q
	    if ($q<$p) {
		# This segment sits entirely in the nonspeech region
		# between the two reference speech segments q and p
 		$falseAlarm{$fileID} += ($segEnd-$segStart);
		$totalFalseAlarm += ($segEnd-$segStart);
	    } else {
		if (($segEnd-$segStart)<0.20) {
		    # For diagnosing Pascal's VAD segmentation
		    print STDOUT "Found short speech region $line\n";
		    ++$numShortSegs;
		} elsif (($segEnd-$segStart)>60.0) {
		    ++$numLongSegs;
		    # For diagnosing Pascal's VAD segmentation
		    print STDOUT "Found long speech region $line\n";
		}
		# There is some overlap with segments p through q
		for ($s=$p; $s<=$q; ++$s) {
		    if ($segStart<$refSegStart[$s]) {
			# There is a leading false alarm portion before s
			$falseAlarm{$fileID} += ($refSegStart[$s]-$segStart);
			$totalFalseAlarm += ($refSegStart[$s]-$segStart);
			$segStart=$refSegStart[$s];
		    }
		    $speechPortion = ($refSegEnd[$s]<$segEnd) ?
			($refSegEnd[$s]-$segStart) : ($segEnd-$segStart);
		    $foundSpeech{$fileID} += $speechPortion;
		    $totalFoundSpeech += $speechPortion;
		    $segStart=$refSegEnd[$s];
		}
		if ($segEnd>$segStart) {
		    # There is a trailing false alarm portion after q
		    $falseAlarm{$fileID} += ($segEnd-$segStart);
		    $totalFalseAlarm += ($segEnd-$segStart);
		}
	    }
	}
    } else {
	# This FileID does not exist in the reference segmentation
	# So all this speech counts as a false alarm
  exit (1);
	printf STDERR ("Unexpected fileID in hypothesized segments: %s", $fileID);
	$totalFalseAlarm += ($segEnd-$segStart);
    }
    ++$numLines;
}
close(SEGMENTS);
print STDERR "Read $numLines segments from $HypothesizedSegmentation\n";

################################################################################
# Now that all hypothesized segments have been processed, compute needed stats
################################################################################

$totalActualSpeech = 0.0;
$totalNonSpeechEst = 0.0; # This is just a crude estimate of total nonspeech.
foreach $fileID (sort keys %actualSpeech) {
    $totalActualSpeech += $actualSpeech{$fileID};
    $totalNonSpeechEst += $maxEndTime{$fileID} - $actualSpeech{$fileID};
    #######################################################################
    # Print file-wise statistics to STDOUT; can pipe to /dev/null is needed
    #######################################################################
    printf STDOUT ("%s: %.2f min actual speech, %.2f min hypothesized: %.2f min overlap (%d\%), %.2f min false alarm (~%d\%)\n",
		   $fileID,
		   ($actualSpeech{$fileID}/60.0),
		   ($hypothesizedSpeech{$fileID}/60.0),
		   ($foundSpeech{$fileID}/60.0),
		   ($foundSpeech{$fileID}*100/($actualSpeech{$fileID}+0.01)),
		   ($falseAlarm{$fileID}/60.0),
		   ($falseAlarm{$fileID}*100/($maxEndTime{$fileID}-$actualSpeech{$fileID}+0.01)));
}

################################################################################
# Finally, we have everything needed to report the segmentation statistics.
################################################################################

printf STDERR ("------------------------------------------------------------------------\n");
printf STDERR ("TOTAL: %.2f hrs actual speech, %.2f hrs hypothesized: %.2f hrs overlap (%d\%), %.2f hrs false alarm (~%d\%)\n",
		   ($totalActualSpeech/3600.0),
		   ($totalHypSpeech/3600.0),
		   ($totalFoundSpeech/3600.0),
		   ($totalFoundSpeech*100/($totalActualSpeech+0.000001)),
		   ($totalFalseAlarm/3600.0),
		   ($totalFalseAlarm*100/($totalNonSpeechEst+0.000001)));
printf STDERR ("\t$numShortSegs segments < 0.2 sec and $numLongSegs segments > 60.0 sec\n");
printf STDERR ("------------------------------------------------------------------------\n");
