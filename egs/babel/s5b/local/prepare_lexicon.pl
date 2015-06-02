#!/usr/bin/env perl
use Getopt::Long;
use Data::Dumper;

###############################################################################
#
# Convert a Babel-formatted dictionary to work with Kaldi, and optionally
# add non-speech "words" that appear in the transcription. e.g. <laughter>
#
# Convert dictionary from entries of the form
#
#       WORD	Romanization	pronunciation1	pronunciation2	...
#
# where each pronunciation has syllable boundaries [.#] and tags _X, " or %
#
#       Phone1 Phone2 _TAG . Phone1 Phone2 Phone3 _TAG
#
# and so on, e.g.
#
#       㓤	gat1	g 6 t _1	h O: t _3	k i: t _1
#       兄妹	hing1mui2	h i: N _1 . m u:j _2	h i: N _1 . m u:j _6
#
# to entries of the form
#
#       㓤	g_1 6_1 t_1
#       㓤	h_3 O:_3 t_3
#       㓤	k_1 i:_1 t_1
#       兄妹	h_1 i:_1 N_1 m_2 u:j_2
#       兄妹	h_1 i:_1 N_1 m_6 u:j_6
#       
#
# Write only one pronunciation per line
# Transfer any tags, prefixed by underscores, to phones in the syllable 
# Remove the syllable boundary markers, given by periods or pound signs
#
# NOTE: The Romainzation is present only for some languages.  See -r option.
#
# This script will create 5 new files
#
#   -  lexicon.txt: words from the original lexicon + some non-speech "words"
#
       $OOV_symbol = "<unk>";     # Default OOV symbol: pronunciation <oov>
       $vocalNoise = "<v-noise>"; # Vocal noise symvol: pronunciation <vns>
       $nVoclNoise = "<noise>";   # Nonvocal noise:     pronunciation <sss>
       $silence    = "<silence>"; # Silence > 1 second: pronunciation $sil
       $icu_transform = "";
       $phonemap="";
#
#   -  nonsilence_phones.txt: tagged phones from the new lexicon 
#
#   -  optional_silence.txt: phones used to model silence in acoustic training
#
       $sil = "SIL"; # Also the pronunciation of the word token $silence
#
#   -  silence_phones.txt: $sil and special phones for non-speech "words"
#
#   -  extra_questions.txt: sets of phones of the form *_TAG, one set per line
#
# The last file provides sets of phones that share a tag, so that questions can
# effectively be asked about the tag of a neighboring phone during clustering.
#
###############################################################################

GetOptions("add=s" => \$nsWordsFile,  
           "oov=s" => \$OOV_symbol, 
           "romanized!" => \$romanized, 
           "sil=s" => \$sil, 
           "icu-transform=s" => \$icu_transform,
           "phonemap=s" => \$phonemap 
           );

if ($#ARGV == 1) {
    $inDict = $ARGV[0];
    $outDir = $ARGV[1];
    print STDERR ("$0: $inDict $outDir\n");
    print STDERR ("\tNon-speech words will be added from $nsWordsFile\n") if ($nsWordsFile);
    print STDERR ("\tUnknown words will be represented by \"$OOV_symbol\"\n") unless ($OOV_symbol eq "<unk>");
    print STDERR ("\tRomanized forms of words expected in the dictionary\n") if ($romanized);
    print STDERR ("\tThe optional silence phone will be \"$OOV_symbol\"\n") unless ($sil eq "SIL");
    print STDERR ("\tThe ICU transform for case-conversion will be: \"$icu_transform\"\n") if ($icu_transform);
} else {
    print STDERR ("Usage: $0 [--options] BabelDictionary OutputDir\n");
    print STDERR ("\t--add <filename>  Add these nonspeech words to lexicon\n");
    print STDERR ("\t--oov <symbol>    Use this symbol for OOV words (default <unk>)\n");
    print STDERR ("\t--romanized       Dictionary contains (omissible) romanized word-forms\n");
    print STDERR ("\t--phonemap <maps> During reading the dictionary, perform the specified \n");
    print STDERR ("\t                  phoneme mapping. The format is: p1=p1' p2' p3';p2=p4'\n");
    print STDERR ("\t                  where p1 and p2 are existing phonemes and p1'..p4' are\n");
    print STDERR ("\t                  either new or existing phonemes\n");
    print STDERR ("\t--icu-transform   ICU transform to be used during the ICU transliteration\n");
    exit(1);
}

unless (-d $outDir) {
    print STDERR ("$0: Creating output directory $outDir\n");
    die "Unable to create output directory $outDir"
	if system("mkdir -p $outDir"); # mkdir returned with status != 0
}
$outLex  = "$outDir/lexicon.txt";
$nspFile = "$outDir/nonsilence_phones.txt";
$spFile  = "$outDir/silence_phones.txt";
$osFile  = "$outDir/optional_silence.txt";
$exqFile = "$outDir/extra_questions.txt";


#The phonemap is in the form of "ph1=a b c;ph2=a f g;...."
%phonemap_hash;
if ($phonemap) {
  $phonemap=join(" ", split(/\s+/, $phonemap));
  print $phonemap . "\n";
  @phone_map_instances=split(/;/, $phonemap);
  foreach $instance (@phone_map_instances) {
    ($phoneme, $tgt) = split(/=/, $instance);
    $phoneme =~ s/^\s+|\s+$//g;
    $tgt =~ s/^\s+|\s+$//g;
    #print "$phoneme=>$tgt\n";
    @tgtseq=split(/\s+/,$tgt);
    $phonemap_hash{$phoneme} = [];
    push @{$phonemap_hash{$phoneme}}, @tgtseq;
  }
}

#print Dumper(\%phonemap_hash);

###############################################################################
# Read input lexicon, write output lexicon, and save the set of phones & tags.
###############################################################################


open (INLEX, $inDict)
    || die "Unable to open input dictionary $inDict";

open (OUTLEX, "| sort -u > $outLex")
    || die "Unable to open output dictionary $outLex";

$numWords = $numProns = 0;
while ($line=<INLEX>) {
    chomp;
    ###############################################
    # Romainzed forms necessitate \t\S+ below, else
    # if ($line =~ m:^([^\t]+)(\t[^\t]+)+$:) {
    ###############################################
    if ( ($romanized && ($line =~ m:^([^\t]+)\t\S+((\t[^\t]+)+)$:)) ||
	 ((!$romanized) && ($line =~ m:^([^\t]+)((\t[^\t]+)+)$:)) ) {
        $word  = $1;

        if ( $icu_transform ) {
          $xform_word=`echo \"$word\" | uconv -f utf8 -t utf8 -x \"$icu_transform\"`;
          chop $xform_word;
          #print $xform_word;
          #$xform_word="[$word]$xform_word";
        } else {
          $xform_word=$word;
        }
        $prons = $2;
        $prons =~ s:^\s+::;           # Remove leading white-space
        $prons =~ s:\s+$::;           # Remove trailing white-space
        @pron  = split("\t", $prons);
        for ($p=0; $p<=$#pron; ++$p) {
            $new_pron = "";
            while ($pron[$p] =~ s:^([^\.\#]+)[\.\#]{0,1}::) { push (@syllables, $1); }
            while ($syllable = shift @syllables) {
                $syllable =~ s:^\s+::;
                $syllable =~ s:\s+$::;
                $syllable =~ s:\s+: :g;
                @original_phones = split(" ", $syllable);
                @substituted_original_phones=();
                
                foreach $phone (@original_phones) {
                  if (defined $phonemap_hash{$phone} ) {
                    #print "Sub: $phone => " . join (' ', @{$phonemap_hash{$phone}}) . "\n";
                    push @substituted_original_phones, @{$phonemap_hash{$phone}};
                  } else {
                    push @substituted_original_phones, $phone;
                  }
                }
                #print join(' ', @original_phones) . "=>" . join(' ',@substituted_original_phones) . "\n";
                @original_phones = @substituted_original_phones;

                $sylTag = "";
                $new_phones = "";
                while ($phone = shift @original_phones) {
                    if ($phone =~ m:^\_\S+:) {
                        # It is a tag; save it for later
                        $is_original_tag{$phone} = 1;
                        $sylTag .= $phone;
                    } elsif ($phone =~ m:^[\"\%]$:) {
                        # It is a stress marker; save it like a tag
                        $phone = "_$phone";
                        $is_original_tag{$phone} = 1;
                        $sylTag .= $phone;
                    } elsif ( $phone =~ m:_:) {
                      # It is a phone containing "_" (underscore)
                      $new_phone=$phone;
                      $new_phone=~ s/\_//g;
                      if (( $is_original_phone{$phone} ) and not defined( $substituted_phones{phone}) ) {
                        die "ERROR, the $new_phone and $phone are both existing phones, so we cannot do automatic map!";
                      } else {
                        print STDERR "WARNING, phone $phone was replaced with $new_phone\n" unless $substituted_phones{$phone};
                      }
                      $is_original_phone{$new_phone} = "$new_phone";
                      $substituted_phones{$phone} = $new_phone;
                      $new_phones .= " $new_phone";
                    } else {
                        # It is a phone
                        if ( $substituted_phones{phone} ) {
                          die "ERROR, the $new_phone and $phone are both existing phones, so we cannot do automatic map!";
                        }                        
                        $is_original_phone{$phone} = "$phone";
                        $new_phones .= " $phone";
                    }
                }
                $new_phones =~ s:(\S+):$1${sylTag}:g;
                $new_pron .= $new_phones . "\t"; # the tab added by Dan, to keep track of
                                                 # syllable boundaries.
                $is_compound_tag{$sylTag} = 1;
                while ($new_phones =~ s:^\s*(\S+)::) { $is_new_phone{$1} = 1; }
            }
            $new_pron =~ s:^\s+::;
            print OUTLEX ("$xform_word\t$new_pron\n");
            $numProns++;
        }
        @pron = ();
        $numWords++;
    } else {
        print STDERR ("$0 WARNING: Skipping unparsable line $. in $inDict\n");
    }
}
close(INLEX)
    && print STDERR ("$0: Read $numWords entries from $inDict\n");

###############################################################################
# Read a list of non-speech words if given, and write their "pronunciations"
#   - Such lexicon entries are typically created for <cough>, <laugh> etc.
#   - If provided explicitly, they each get their own private phone models
#   - Otherwise, they are mapped to an OOV symbol with a shared <oov> phone
#   - All such phones are grouped with the $sil phone for clustering purposes,
#     which means that they remain context-independent and form a question set.
###############################################################################

if ($nsWordsFile) {
    open (NSW, $nsWordsFile)
        || die "Unable to open non-speech words file $nsWordsFile";
    $numNSWords = 0;
    while ($line=<NSW>) {
        next unless ($line =~ m:^\s*([^\s]+)\s*:); # Take the first word if present
        print OUTLEX ("$1\t$1\n");                 # The word itself is its pronunciation
        $is_silence_phone{$1} = 1;                 # Add it to the list of silence phones
        $numProns++;
        $numNSWords++;
    }
    close(NSW)
        && print STDERR ("$0: Adding $numNSWords non-speech words from $nsWordsFile to $outLex\n");
}

# Add the OOV symbol to the lexicon
print OUTLEX ("$OOV_symbol\t<oov>\n");             # The symbol <oov> is assumed not to be
$is_silence_phone{"<oov>"} = 1;                    # a phone in the original lexicon :-)
$numProns++;

# Add the vocal noise symbol to the lexicon
print OUTLEX ("$vocalNoise\t<vns>\n");             # The symbol <vns> is assumed not to be
$is_silence_phone{"<vns>"} = 1;                    # a phone in the original lexicon :-)
$numProns++;

# Add the nonvocal noise symbol to the lexicon
print OUTLEX ("$nVoclNoise\t<sss>\n");             # The symbol <sss> is assumed not to be
$is_silence_phone{"<sss>"} = 1;                    # a phone in the original lexicon :-)
$numProns++;

# Finally, add the silence symbol to the lexicon
print OUTLEX ("$silence\t$sil\n");
$is_silence_phone{$sil} = 1;
$numProns++;

close(OUTLEX)
    && print STDERR ("$0: Wrote $numProns pronunciations to $outLex\n");

###############################################################################
#   -  nonsilence_phones.txt: tagged phones from the new lexicon, 1 phone/line 
###############################################################################

foreach $phone (sort keys %is_new_phone) {
    $tagLess_phone = $phone;
    $tagLess_phone =~ s:^([^_]+).*:$1:; # underscore marks tag beginnings
    if ($is_original_phone{$tagLess_phone}) {
        # save $phone for writing later to the NSP file
        $is_original_phone{$tagLess_phone} .= " $phone";
    } else {
        print STDERR ("$0 WARNING: Skipping unexpected tagged phone $phone.\n");
        print STDERR ("\tCheck if original lexicon has phones containing \"\_\"\n");
        die "Cannot continue";
    }
}

open (NSP, "| sort > $nspFile")
    || die "Unable to write nonsilence phones to $nspFile";
$p = 0;
foreach $phone (sort keys %is_original_phone) {
    $tagged_phones = $is_original_phone{$phone};
    $tagged_phones =~ s:^\S+\s*::;      # Remove the original (untagged) phone
    unless ($phone eq "") {
        print NSP ("$tagged_phones\n"); # Write out the remaining (tagged) phones
        $p++;
    }
}

close(NSP)
    && print STDERR ("$0: Wrote $p (sets of) nonsilence phones to $nspFile\n");

if ( $p > (0.5*$numWords) ) {
    print STDERR ("$0 WARNING: Original dictionary had $numWords words, and\n");
    print STDERR ("\t\t$p nonspeech phones were found!  This is highly unusual.\n");
    print STDERR ("\t\tCheck if the dictionary contains other tab-separated values\n");
    print STDERR ("\t\tthat are being mistaken for pronunciations by this script.\n");
    print STDERR ("$0 ADVICE: Use --romanized for omitting romanized word forms\n") unless ($romanized);
}

###############################################################################
#   -  silence_phones.txt: $sil and special phones for non-speech "words"
###############################################################################

open (SPF, "| sort > $spFile")
    || die "Unable to write silence phones to $spFile";
$p = 0;
foreach $phone (keys %is_silence_phone) {
    print SPF ("$phone\n");
    $p++;
}
close(SPF)
    && print STDERR ("$0: Wrote $p silence phones to $spFile\n");

###############################################################################
#   -  optional_silence.txt: the reserved (?) phone <silence>
###############################################################################

$is_optional_silence{$sil} = 1;
open (OSF, "| sort > $osFile")
    || die "Unable to write optional silence phones to $osFile";
$p = 0;
foreach $phone (keys %is_optional_silence) {
    print OSF ("$phone\n");
    $p++;
}
close(OSF)
    && print STDERR ("$0: Wrote $p optional silence phones to $osFile\n");

###############################################################################
#   -  extra_questions.txt: sets of phones of the form *_TAG, one set per line
###############################################################################

open (EXQ, "| sort > $exqFile")
    || die "Unable to write the extra questions file $exqFile";

# First make sets of all tagged phones that share the (single) original tags

$numExtraQs = 0;
foreach $tag (sort keys %is_original_tag) {
    $question = "";
    foreach $phone (sort keys %is_new_phone) {
        $question .= " $phone" if ($phone =~ m:$tag:);
    }
    $question =~ s:^\s+::;
    print EXQ ("$question\n") unless ($question eq "");
    $numExtraQs++;
}
print STDERR ("$0: Found $numExtraQs unique individual tags in $inDict\n");

# It is possible to go overboard by creating questions with all 2^K possible
# subsets of the original tags. E.g. ($phone=~m:$tag1:)||($phone=~m:$tag2:)
# Do this by hand if it is linguistically meaningful for some language
# It is not worth doing this generically for all languages and tag sets.

# If each syllable has only one tag, then questions with conjunctions of tags
# such as ($phone=~m:$tag1:)&&($phone=~m:$tag2:) will yield empty questions
# However, if syllables carry multiple tags, e.g. tone and stress, then one
# could similarly go overboard with conjunctions of overlapping tags.
# This too is not worth doing generically for all languages and tag sets.

# Instead, just make sets of all tagged phones with the same new (compound) tag

foreach $tag (sort keys %is_compound_tag) {
    next if ($is_original_tag{$tag});
    $question = "";
    foreach $phone (sort keys %is_new_phone) {
        $question .= " $phone" if ($phone =~ m:$tag:);
    }
    $question =~ s:^\s+::;
    print EXQ ("$question\n") unless ($question eq "");
    $numExtraQs++;
}

# Finally, add the silence phones as a set for use as a clustering question

$question = "";
foreach $phone (sort keys %is_silence_phone) {
    $question .= " $phone";
}
$question =~ s:^\s+::;
print EXQ ("$question\n") unless ($question eq "");
$numExtraQs++;

close(EXQ)
    && print STDERR ("$0: Wrote $numExtraQs extra questions (incl compound tags and sil) to $exqFile\n");
