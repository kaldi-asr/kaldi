#!/usr/bin/env perl
use Getopt::Long;

########################################################################
#
# Script to prepare the Babel acoustic training data for Kaldi.
#
#  -  Place transcripts in a file named "text"
#     Each line contains: utteranceID word1 word2 ...
#
#  -  Place the utterance-to-speaker map in a file named "utt2spk"
#     Each line contains: utteranceID speakerID
#     speakerID MUST BE be a prefix of the utteranceID
#     Kaldi code does not require it, but some training scripts do.
#
#   -  Place the utterance-to-segment map in a file named "segments"
#      Each line contains: utteranceID recordingID startTime endTime
#
#   -  Place the recordingID-to-waveformFile map in "wav.scp"
#      Each line contains: recordingIB Input_pipe_for_reading_waveform|
#
#  -  Place the speaker-utterance map in a file named "spk2utt"
#     Each line contains: speakerID utteranceID_1 utteranceID_2 ...
#     This is the inverse of the utt2spk mapping
#
# Note 1: the utteranceIDs in the first 3 files must match exactly, and
#         the recordingIDSs in the last 2 files must match exactly.
#
# Note 2: Babel data formats and file-naming conventions are assumed.
#
#   -  The transcriptions and waveforms are in subdirectories named
#        audio/<filename>.sph
#        transcription/<filename>.txt
#      There is 1 pair of files per recording, with extensions as above
#
#   -  The audio is in NIST sphere format, so shp2pipe may be used, e.g.
#        BABEL_BP_101_11694_20111204_205320_inLine \
#        /export/babel/sanjeev/kaldi-trunk/tools/sph2pipe_v2.5/sph2pipe \
#        -f wav -p -c 1 \
#        BABEL_BP_101_11694_20111204_205320_inLine.sph|
#
#   -  The filename contains speaker information, e.g.
#        BABEL_BP_101_37210_20111102_170037_O1_scripted.sph -> 37210_A
#        BABEL_BP_101_37210_20111102_172955_inLine.sph      -> 37210_A
#        BABEL_BP_101_37210_20111102_172955_outLine.sph     -> 37210_B
#      Specifically, the inLine speaker is the same as scripted
#
#   -  The transcription file has time marks in square brackets, e.g.
#        [0.0]
#        <no-speech> 喂 <no-speech>
#        [7.05]
#        啊 听 听唔听到 啊 <no-speech> 你 而家 仲未 上课 系 嘛 <no-speech>
#        [14.07]
#
#  -  If a vocabulary is provided, map all OOV tokens to an OOV symbol,
#     and write out an OOV list with counts to a file named "oovCounts"
#
#     If one or more word-fragment markers are provided, this script
#     checks if an OOV token can be made in-vocabulary by stripping off
#     the markers one  by one from either end of the token.
#
#     The default settings are
#
      $vocabFile = "";       # No vocab file; nothing is mapped to OOV
      $OOV_symbol = "<unk>"; # Default OOV symbol
      $fragMarkers = "";     # No characters are word-fragment markers
#
#  -  Babel transcriptions contain 4 kinds of untranscribed words
#
#         (())         designates unintelligible words
#         <foreign>    designates a word in another language
#         <prompt>     designates a sequence of pre-recorded words
#         <overlap>    designates two simultaneous foreground speakers
#
#     This script maps them to OOV.  They are not included in oovCounts
#
#  -  Babel transcriptions also contain a few non-linguistics tokens
#
#         <limspack>   map to a vocal noise symbol
#         <breath>     map to a vocal noise symbol
#         <cough>      map to a vocal noise symbol
#         <laugh>      map to a vocal noise symbol
#
#         <click>      map to a nonvocal noise symbol
#         <ring>       map to a nonvocal noise symbol
#         <dtmf>       map to a nonvocal noise symbol
#         <int>        map to a nonvocal noise symbol
#
#         <no-speech>  designates silence > 1 sec.
#
      $vocalNoise = "<v-noise>";
      $nVoclNoise = "<noise>";
      $silence    = "<silence>";
      $icu_transform="";
      $get_whole_transcripts = "false";
#
########################################################################

print STDERR "$0 " . join(" ", @ARGV) . "\n";
GetOptions("fragmentMarkers=s" => \$fragMarkers,
           "oov=s" => \$OOV_symbol,
           "vocab=s" => \$vocabFile,
           "icu-transform=s" => \$icu_transform,
           "get-whole-transcripts=s" => \$get_whole_transcripts
           );

if ($#ARGV == 1) {
    $inDir  = $ARGV[0];
    $outDir = $ARGV[1];
    print STDERR ("$0: $inDir $outDir\n");
    if($vocabFile) {
	print STDERR ("\tLimiting transcriptions to words in $vocabFile\n");
	print STDERR ("\tMapping OOV tokens to \"$OOV_symbol\"\n");
	print STDERR ("\tif they remain OOV even after removing [$fragMarkers] from either end\n") if ($fragMarkers);
    }
    print STDERR ("$0 ADVICE: Use full path for the Input Directory\n") unless ($inDir=~m:^/:);
} else {
    print STDERR ("Usage: $0 [--options] InputDir OutputDir\n");
    print STDERR ("\t--vocab <file>             File containing the permitted vocabulary\n");
    print STDERR ("\t--oov <symbol>             Use this symbol for OOV words (default <unk>)\n");
    print STDERR ("\t--fragmentMarkers <chars>  Remove these from ends of words to minimize OOVs (default none)\n");
    print STDERR ("\t--get-whole-transcripts (true|false) Do not remove utterances containing no speech\n");
    exit(1);
}

########################################################################
# Read and save the vocabulary and map anything not in the vocab <unk>
########################################################################

if ($vocabFile) {
    open (VOCAB, $vocabFile)
        || die "Unable to open vocabulary file $vocabFile";
    $numWords = 0;
    while (<VOCAB>) {
        next unless (m:^([^\s]+):);
        $numWords++ unless (exists $inVocab{$1}); # Don't count word repetitions
        $inVocab{$1} = 1;                         # commonly found in lexicons
    }
    close(VOCAB);
    print STDERR ("Read $numWords unique words from $vocabFile\n");
}

########################################################################
# First read segmentation information from all the transcription files
########################################################################

$TranscriptionDir = "$inDir/transcription";
if (-d $TranscriptionDir) {
    @TranscriptionFiles = `ls ${TranscriptionDir}/*.txt`;
    if ($#TranscriptionFiles >= 0) {
        printf STDERR ("$0: Found %d .txt files in $TranscriptionDir\n", ($#TranscriptionFiles +1));
        $numFiles = $numUtterances = $numWords = $numOOV = $numSilence = 0;
        while ($filename = shift @TranscriptionFiles) {
            $fileID =  $filename;     # To capture the base file name
            $fileID =~ s:.+/::;       # remove path prefix
            $fileID =~ s:\.txt\s*$::; # remove file extension
            # For each transcription file, extract and save segmentation data
            $numUtterancesThisFile = 0;
            $prevTimeMark = -1.0;
            $text = "";
            if ( $icu_transform ) {
              $inputspec="uconv -f utf8 -t utf8 -x \"$icu_transform\" $filename |";
            } else {
              $inputspec=$filename;
            }
            open (TRANSCRIPT, $inputspec) || die "Unable to open $filename";
            while ($line=<TRANSCRIPT>) {
                chomp $line;
                if ($line =~ m:^\s*\[([0-9]+\.*[0-9]*)\]\s*$:) {
                    $thisTimeMark = $1;
                    if ($thisTimeMark < $prevTimeMark) {
                      print STDERR ("$0 ERROR: Found segment with negative duration in $filename\n");
                      print STDERR ("\tStart time = $prevTimeMark, End time = $thisTimeMark\n");
                      print STDERR ("\tThis could be a sign of something seriously wrong!\n");
                      print STDERR ("\tFix the file by hand or remove it from the directory, and retry.\n");
                      exit(1);
                    }
                    if ($prevTimeMark<0) {
                        # Record the first timemark and continue
                        $prevTimeMark = $thisTimeMark;
                        next;
                    }
                    ##################################################
                    # Create an utteranceID using fileID & start time
                    #    -  Assume Babel file naming conventions
                    #    -  Remove prefix: program_phase_language
                    #    -  inLine = scripted = spkr A, outLine = B
                    #    -  Move A/B so that utteranceIDs sort by spkr
                    #    -  Assume utterance start time < 10000 sec.
                    ##################################################
                    $utteranceID =  $fileID;
                    $utteranceID =~ s:[^_]+_[^_]+_[^_]+_::;
                    $utteranceID =~ s:([^_]+)_(.+)_(inLine|scripted):${1}_A_${2}:;
                    $utteranceID =~ s:([^_]+)_(.+)_outLine:${1}_B_${2}:;
                    $utteranceID .= sprintf ("_%06i", (100*$prevTimeMark));
                    ##################################################
                    # Then save segmentation, transcription, spkeaerID
                    ##################################################
                    if (exists $transcription{$utteranceID}) {
                        # utteranceIDs should be unique, but this one is not!
                        # Either time marks in the transcription file are bad,
                        # or something went wrong in generating the utteranceID
                        print STDERR ("$0 WARNING: Skipping duplicate utterance $utteranceID\n");
                    }
                    elsif ($text eq "") {
                        # Could be due to text filtering done below
                        # Output information to STDOUT to enable > /dev/null
                        print STDOUT ("$0: Skipping empty transcription $utteranceID\n");
                    } else {
                        $transcription{$utteranceID} = $text;
                        $startTime{$utteranceID} = $prevTimeMark;
                        $endTime{$utteranceID} = $thisTimeMark;
                        if ($utteranceID =~ m:([^_]+_[AB]).*:) {
                            $speakerID{$utteranceID} = $1;
                        } else {
                            # default: one speaker per audio file
                            $speakerID{$utteranceID} = $fileID;
                        }
                        $baseFileID{$utteranceID} = $fileID;
                        $numUtterancesThisFile++;
                        $numUtterances++;
                        $text = "";
                    }
                    $prevTimeMark = $thisTimeMark;
                } else {
		    @tokens = split(/\s+/, $line);
		    $text = "";
		    while ($w = shift(@tokens)) {
			# First, some Babel-specific transcription filtering
			if (($w eq "<sta>")||($w eq "<male-to-female>")||($w eq "<female-to-male>")||($w eq "~")) {
			    next;
			} elsif (($w eq "<lipsmack>")||($w eq "<breath>")||($w eq "<cough>")||($w eq "<laugh>")) {
			    $text .= " $vocalNoise";
			    $numWords++;
			} elsif (($w eq "<click>")||($w eq "<ring>")||($w eq "<dtmf>")||($w eq "<int>")){
			    $text .= " $nVoclNoise";
			    $numWords++;
			} elsif (($w eq "(())")||($w eq "<foreign>")||($w eq "<overlap>")||($w eq "<prompt>")) {
			    $text .= " $OOV_symbol";
			    $oovCount{$w}++;
			    $numOOV++;
			    $numWords++;
			} elsif ($w eq "<no-speech>") {
			    $text .= " $silence";
			    $numSilence++;
			} else {
			    # This is a just regular spoken word
			    if ($vocabFile && (! $inVocab{$w}) && $fragMarkers) {
            print "Not in vocab: $w\n";
				# $w is a potential OOV token
				# Remove fragMarkers to see if $w becomes in-vocabulary
				while ($w =~ m:^(\S+[$fragMarkers]|[$fragMarkers]\S+)$:) {
				    if ($w =~ m:^(\S+)[$fragMarkers]$:) {
					$w = $1;
					last if ($inVocab{$w});
				    } elsif ($w =~m:^[$fragMarkers](\S+)$:) {
					$w = $1;
					last if ($inVocab{$w});
				    } else {
					die "Logically, the program should never reach here!";
				    }
				}
			    }
			    # If still an OOV, replace $w by $OOV_symbol
			    if ($vocabFile && (! $inVocab{$w})) {
				# $w is definitely an OOV token
				if (exists $oovCount{$w}) {
				    $oovCount{$w}++;
				} else {
				    $oovCount{$w} = 1;
				}
				$w = $OOV_symbol;
				$numOOV++;
			    }
			    $text .= " $w";
			    $numWords++;
			}
		    }
		    $text =~ s:^\s+::; # Remove leading white space, if any
        # Transcriptions must contain real words to be useful in training
        if ($get_whole_transcripts ne "true") {
          $text =~ s:^(($OOV_symbol|$vocalNoise|$nVoclNoise|$silence)[ ]{0,1})+$::;
        }
		}
	    }
            close(TRANSCRIPTION);
            if ($numUtterancesThisFile>0) {
                $lastTimeMarkInFile{$fileID} = $prevTimeMark;
                $numUtterancesInFile{$fileID} = $numUtterancesThisFile;
                $numUtterancesThisFile = 0;
            }
            $numFiles++;
        }
        print STDERR ("$0: Recorded $numUtterances non-empty utterances from $numFiles files\n");
    } else {
        print STDERR ("$0 ERROR: No .txt files found $TranscriptionDir\n");
        exit(1);
    }
} else {
    print STDERR ("$0 ERROR: No directory named $TranscriptionDir\n");
    exit(1);
}

########################################################################
# Then verify existence of corresponding audio files and their durations
########################################################################

$AudioDir = "$inDir/audio";
if (-d $AudioDir) {
    @AudioFiles = `ls ${AudioDir}/*.sph`;
    if ($#AudioFiles >= 0) {
        printf STDERR ("$0: Found %d .sph files in $AudioDir\n", ($#AudioFiles +1));
        $numFiles = 0;
        while ($filename = shift @AudioFiles) {
            $fileID = $filename;
            $fileID =~ s:.+/::;      # remove path prefix
            $fileID =~ s:\.sph\s*::; # remove file extension
            if (exists $numUtterancesInFile{$fileID}) {
                # Some portion of this file has training transcriptions
                @Info = `head $filename`;
                $SampleCount = -1;
                $SampleRate  = 8000; #default
                while ($#Info>=0) {
                   $line = shift @Info;
                   $SampleCount = $1 if ($line =~ m:sample_count -i (\d+):);
                   $SampleRate  = $1 if ($line =~ m:sample_rate -i (\d+):);
                }
                if ($SampleCount<0) {
                    # Unable to extract a valid duration from the sphere header
                    print STDERR ("Unable to extract duration: skipping file $filename");
                } else {
                    $waveformName{$fileID} = $filename; chomp $waveformName{$fileID};
                    $duration{$fileID} = $SampleCount/$SampleRate;
                    $numFiles++;
                }
            } else {
                # Could be due to text filtering resulting in an empty transcription
                # Output information to STDOUT to enable > /dev/null
                print STDOUT ("$0: No transcriptions for audio file ${fileID}.sph\n");
            }
        }
        print STDERR ("$0: Recorded durations from headers of $numFiles .sph files\n");
    } else {
        print STDERR ("$0 NOTICE: No .sph files in $AudioDir\n");
    }

    @AudioFiles = `ls ${AudioDir}/*.wav`;
    if ($#AudioFiles >= 0) {
        $soxi=`which soxi` or die "$0: Could not find soxi binary -- do you have sox installed?\n";
        chomp $soxi;
        printf STDERR ("$0: Found %d .wav files in $AudioDir\n", ($#AudioFiles +1));
        $numFiles = 0;
        while ($filename = shift @AudioFiles) {
            $fileID = $filename;
            $fileID =~ s:.+/::;      # remove path prefix
            $fileID =~ s:\.wav\s*::; # remove file extension
            if (exists $numUtterancesInFile{$fileID}) {
                # Some portion of this file has training transcriptions
                $duration = `$soxi -D $filename`;
                if ($duration <=0) {
                    # Unable to extract a valid duration from the sphere header
                    print STDERR ("Unable to extract duration: skipping file $filename");
                } else {
                    if (exists $waveformName{$fileID} ) {
                      print STDERR ("$0 ERROR: duplicate fileID \"$fileID\" for files \"$filename\" and \"" . $waveformName{$fileID} ."\"\n");
                      exit(1);
                    }
                    $waveformName{$fileID} = $filename; chomp $waveformName{$fileID};
                    $duration{$fileID} = $duration;
                    $numFiles++;
                }
            } else {
                # Could be due to text filtering resulting in an empty transcription
                # Output information to STDOUT to enable > /dev/null
                print STDOUT ("$0: No transcriptions for audio file ${fileID}.sph\n");
            }
        }
        print STDERR ("$0: Recorded durations from headers of $numFiles .sph files\n");
    } else {
        print STDERR ("$0 NOTICE: No .wav files in $AudioDir\n");
    }

    if ( $#waveformName == 0 ) {
      print STDERR ("$0 ERROR: No audio files found!");
    }
} else {
    print STDERR ("$0 ERROR: No directory named $AudioDir\n");
    exit(1);
}

########################################################################
# Now all the needed information is available.  Write out the 4 files.
########################################################################

unless (-d $outDir) {
    print STDERR ("$0: Creating output directory $outDir\n");
    die "Failed to create output directory" if (`mkdir -p $outDir`); # i.e. if the exit status is not zero.
}
print STDERR ("$0: Writing 5 output files to $outDir\n");

$textFileName = "$outDir/text";
open (TEXT, "> $textFileName") || die "$0 ERROR: Unable to write text file $textFileName\n";

$utt2spkFileName = "$outDir/utt2spk";
open (UTT2SPK, "> $utt2spkFileName") || die "$0 ERROR: Unable to write utt2spk file $utt2spkFileName\n";

$segmentsFileName = "$outDir/segments";
open (SEGMENTS, "> $segmentsFileName") || die "$0 ERROR: Unable to write segments file $segmentsFileName\n";

$scpFileName = "$outDir/wav.scp";
open (SCP, "| sort -u >  $scpFileName") || die "$0 ERROR: Unable to write wav.scp file $scpFileName\n";
my $binary=`which sph2pipe` or die "Could not find the sph2pipe command"; chomp $binary;
$SPH2PIPE ="$binary -f wav -p -c 1";
my $SOXBINARY =`which sox` or die "Could not find the sph2pipe command"; chomp $SOXBINARY;
$SOXFLAGS ="-r 8000 -c 1 -b 16 -t wav - downsample";

$spk2uttFileName = "$outDir/spk2utt";
open (SPK2UTT, "> $spk2uttFileName") || die "$0 ERROR: Unable to write spk2utt file $spk2uttFileName\n";

$oovFileName = "$outDir/oovCounts";
open (OOV, "| sort -nrk2 > $oovFileName") || die "$0 ERROR: Unable to write oov file $oovFileName\n";

$numUtterances = $numSpeakers = $numWaveforms = 0;
$totalSpeech = $totalSpeechSq = 0.0;
foreach $utteranceID (sort keys %transcription) {
    $fileID = $baseFileID{$utteranceID};
    if (exists $waveformName{$fileID}) {
        # There are matching transcriptions and audio
        $numUtterances++;
      	$totalSpeech += ($endTime{$utteranceID} - $startTime{$utteranceID});
        $totalSpeechSq += (($endTime{$utteranceID} - $startTime{$utteranceID})
			   *($endTime{$utteranceID} - $startTime{$utteranceID}));
        print TEXT ("$utteranceID $transcription{$utteranceID}\n");
        print UTT2SPK ("$utteranceID $speakerID{$utteranceID}\n");
        print SEGMENTS ("$utteranceID $fileID $startTime{$utteranceID} $endTime{$utteranceID}\n");
        if (exists $uttList{$speakerID{$utteranceID}}) {
            $uttList{$speakerID{$utteranceID}} .= " $utteranceID";
        } else {
            $numSpeakers++;
            $uttList{$speakerID{$utteranceID}} = "$utteranceID";
        }
        next if (exists $scpEntry{$fileID});
        $numWaveforms++;
        if ($waveformName{$fileID} =~ /.*\.sph/ ) {
          $scpEntry{$fileID} = "$SPH2PIPE $waveformName{$fileID} |";
        } else {
          $scpEntry{$fileID} = "$SOXBINARY $waveformName{$fileID} $SOXFLAGS |";
        }
    } else {
        print STDERR ("$0 WARNING: No audio file for transcription $utteranceID\n");
    }
}
foreach $fileID (sort keys %scpEntry) {
    print SCP ("$fileID $scpEntry{$fileID}\n");
}
foreach $speakerID (sort keys %uttList) {
    print SPK2UTT ("$speakerID $uttList{$speakerID}\n");
}
foreach $w (sort keys %oovCount) {
    print OOV ("$w\t$oovCount{$w}\n");
}
exit(1) unless (close(TEXT) && close(UTT2SPK) && close(SEGMENTS) && close(SCP) && close(SPK2UTT) && close(OOV));

print STDERR ("$0: Summary\n");
print STDERR ("\tWrote $numUtterances lines each to text, utt2spk and segments\n");
print STDERR ("\tWrote $numWaveforms lines to wav.scp\n");
print STDERR ("\tWrote $numSpeakers lines to spk2utt\n");
print STDERR ("\tHmmm ... $numSpeakers distinct speakers in this corpus? Unusual!\n")
    if (($numSpeakers<($numUtterances/500.0)) || ($numSpeakers>($numUtterances/2.0)));
print STDERR ("\tTotal # words = $numWords (including $numOOV OOVs) + $numSilence $silence\n")
    if ($vocabFile);
printf STDERR ("\tAmount of speech = %.2f hours (including some due to $silence)\n", $totalSpeech/3600.0);
if ($numUtterances>0) {
    printf STDERR ("\tAverage utterance length = %.2f sec +/- %.2f sec, and %.2f words\n",
		   $totalSpeech /= $numUtterances,
		   sqrt(($totalSpeechSq/$numUtterances)-($totalSpeech*$totalSpeech)),
		   $numWords/$numUtterances);
}

exit(0);

########################################################################
# Done!
########################################################################
