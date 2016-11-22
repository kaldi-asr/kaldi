#!/usr/bin/perl -w
use strict;

my $version = "22";

#################################
# History:
#
# version 22:  * JGF: added an option '-m FILE' to hold a CSV speaker map file.
#
# version 21:  * JGF: added a flag '-n' to not remove the directory paths from the source
#                files in the UEM file.
#
# version 20:  * change metadata discard rule:  rather than discard if the midpoint
#                (or endpoint) of the metadata object lies in a no-eval zone, discard
#                if there is ANY overlap whatsoever between the metadata object and
#                a no-eval zone.  This holds for system output objects only if the
#                system output metadata object is not mapped to a ref object.
#              * optimize IP and SU mapping by giving a secondary bonus mapping score
#                to candidate ref-sys MD map pairs if the end-words of both coincide.
#
# version 19:  * bug fix in subroutine speakers_match
#              * bug fix in tag_ref_words_with_metadata_info
#
# version 18:  * cosmetic fix to error message in eval_condition
#              * added conditional output options for word coverage performance
#              * added secondary MD word coverage optimization to word alignment
#              * further optimize word alignment by considering MD subtypes
#              * further optimize MD alignment by considering MD subtypes
#              * add a new SU discard rule:  discard if TEND in no-eval zone
#              * enforce legal values for su_extent_limit
#
# version 17:  create_speaker_segs modified to accommodate the same speaker
#              having multiple overlapping speaker segments.  (This is an
#              error and pathological condition, but the system must either
#              disallow (abort on) the condition, or perform properly under
#              the pathological condition.  The second option is chosen.)
#
# version 16:  * If neither -w nor -W is specified, suppress warnings about
#                ref SPEAKER records subsuming no lexemes.
#              * Output the overall speaker diarization stats after the
#                stats for the individual files
#              * Do not alter the case of alphabetic characters in the filename
#                field from the ref rttm file
#              * Made the format of the overall speaker error line more similar to
#                the corresponding line of output from SpkrSegEval, to facilitate
#                use of existing "grep" commands in existing scripts.
#
# version 15:  * bug fix in create_speaker_segs to accommodate
#                contiguous same-speaker segments
#              * added conditional file/channel scoring to
#                speaker diarization evaluation
#
# version 14:  bug fix in md_score
#
# version 13:  add DISCOURSE_RESPONSE as a FILLER subtype
#
# version 12:  make REF LEXEMES optional if they aren't required
#
# version 11:  change default for noscore MD regions
#
# version 10:  bug fix
#
# version 09:
#    * avoid crash when metadata discard yields no metadata
#    * make evaluated ref_wds sensitive to metadata type
#    * defer discarding of system output metadata until after
#      metadata mapping, then discard only unmapped events.
#    * extend 1-speaker scoring inhibition to metadata
#    * eliminate demand for SPKR-INFO subtype for speakers
#    * correct ref count of IP and SU exact boundary words
#    * add official RT-04F scores
#    * add conditional analyses for file/chnl/spkr/gender
#
# version 08:
#    * bug fixes speaker diarization scoring
#      - count of EVAL_WORDS corrected
#      - no-score extended to nearest SPEAKER boundary
#
# version 07:
#    * warning issued when discarding metadata events
#      that cover LEXEMEs in the evaluation region
#
# version 06:
#    * eliminated unused speakers from speaker scoring
#    * changed discard algorithm for unannotated SU's and
#      complex EDIT's to discard sys SU's and EDIT's when
#      their midpoints overlap (rather than ANY overlap).
#    * fixed display_metadata_mapping
#
# version 05:
#    * upgraded display_metadata_mapping
#
# version 04:
#    * diagnostic metadata mapping output added
#    * uem_from_rttm bug fix
#
# version 03:
#    * adjusted times used for speaker diarization
#    * changed usage of max_extend to agree with cookbook
#
# version 02: speaker diarization evaluation added
#
# version 01: a merged version of df-eval-v14 and su-eval-v16
#
#################################

#global data
my $epsilon = 1E-8;
my $miss_name = "  MISS";
my $fa_name = "  FALSE ALARM";
my %rttm_datatypes = (SEGMENT        => {eval => 1, "<na>" => 1},
		      NOSCORE        => {"<na>" => 1},
		      NO_RT_METADATA => {"<na>" => 1},
		      LEXEME         => {lex => 1, fp => 1, frag => 1, "un-lex" => 1,
					 "for-lex" => 1, alpha => 1, acronym => 1,
					 interjection => 1, propernoun => 1, other => 1},
		      "NON-LEX"      => {laugh => 1, breath => 1, lipsmack => 1,
					 cough => 1, sneeze => 1, other => 1},
		      "NON-SPEECH"   => {noise => 1, music => 1, other => 1},
		      FILLER         => {filled_pause => 1, discourse_marker => 1,
					 discourse_response => 1, explicit_editing_term => 1,
					 other => 1},
		      EDIT           => {repetition => 1, restart => 1, revision => 1,
					 simple => 1, complex => 1, other => 1},
		      IP             => {edit => 1, filler => 1, "edit&filler" => 1,
					 other => 1},
		      SU             => {statement => 1, backchannel => 1, question => 1,
					 incomplete => 1, unannotated => 1, other => 1},
		      CB             => {coordinating => 1, clausal => 1, other => 1},
		      "A/P"          => {"<na>" => 1},
		      SPEAKER        => {"<na>" => 1},
		      "SPKR-INFO"    => {adult_male => 1, adult_female => 1, child => 1, unknown => 1});
my %md_subtypes = (FILLER => $rttm_datatypes{FILLER},
		   EDIT   => $rttm_datatypes{EDIT},
		   IP     => $rttm_datatypes{IP},
		   SU     => $rttm_datatypes{SU});
my %spkr_subtypes = (adult_male => 1, adult_female => 1, child => 1, unknown => 1);

my $noeval_mds = {
    DEFAULT => {
	NOSCORE        => {"<na>" => 1},
	NO_RT_METADATA => {"<na>" => 1},
    },
};
my $noscore_mds = {
    DEFAULT => {
	NOSCORE        => {"<na>" => 1},
	LEXEME         => {"un-lex" => 1},
	SU             => {unannotated => 1},
    },
    MIN => {
	NOSCORE        => {"<na>" => 1},
	SU             => {unannotated => 1},
    },
    FRAG_UNLEX => {
	NOSCORE        => {"<na>" => 1},
	LEXEME         => {frag => 1, "un-lex" => 1},
	SU             => {unannotated => 1},
    },
    FRAG => {
	NOSCORE        => {"<na>" => 1},
	LEXEME         => {frag => 1},
	SU             => {unannotated => 1},
    },
    NONE => {
    },
};
my $noeval_sds = {
    DEFAULT => {
	NOSCORE        => {"<na>" => 1},
    },
};
my $noscore_sds = {
    DEFAULT => {
	NOSCORE        => {"<na>" => 1},
	"NON-LEX"      => {laugh => 1, breath => 1, lipsmack => 1,
			   cough => 1, sneeze => 1, other => 1},
    },
};

my %speaker_map;

my $default_extend = 0.50; #the maximum time (in seconds) to extend a no-score zone
my $default_collar = 0.00; #the no-score collar (in +/- seconds) to attach to SPEAKER boundaries
my $default_tgap = 1.00; #the max gap (in seconds) between matching ref/sys words
my $default_Tgap = 1.00; #the max gap (in seconds) between matching ref/sys metadata events
my $default_Wgap = 0.10; #the max gap (in words) between matching ref/sys metadata events
my $default_su_time_limit = 0.50; #the max extent (in seconds) to match for SU's
my $default_su_word_limit = 2.00; #the max extent (in words) to match for SU's
my $default_word_delta_score = 10.0; #the max delta score for word-based DP alignment of ref/sys words
my $default_time_delta_score = 1.00; #the max delta score for time-based DP alignment of ref/sys words

my $usage = "\n\nUsage: $0 [-h] -r <ref_file> -s <src_file>\n\n".
    "Description:  md-eval evaluates EARS metadata detection performance\n".
    "      by comparing system metadata output data with reference data\n".
    "INPUT:\n".
    "  -R <ref-list>  A file containing a list of the reference metadata files\n".
    "       being evaluated, in RTTM format.  If the word-mediated alignment\n".
    "       option is used then this data must include reference STT data\n".
    "       in addition to the metadata being evaluated.\n".
    "  OR\n".
    "  -r <ref-file>  A file containing reference metadata, in RTTM format\n\n".
    "  -S <sys-list>  A file containing a list of the system output metadata\n".
    "       files to be evaluated, in RTTM format.  If the word-mediated\n".
    "       alignment option is used then this data must include system STT\n".
    "       output data in addition to the metadata to be evaluated.\n".
    "  OR\n".
    "  -s <sys-file>  A file containin system output metadata, in RTTM format\n\n".
    "  input options:\n".
    "    -x to include complex edits in the analysis and scoring.\n".
    "    -w for word-mediated alignment.\n".
    "       * The default (time-mediated) alignment aligns ref and sys metadata\n".
    "         according to the time overlap of the original ref and sys metadata\n".
    "         time intervals.\n".
    "       * Word-mediated alignment aligns ref and sys metadata according to\n".
    "         the alignment of the words that are subsumed within the metadata\n".
    "         time intervals.\n".
    "    -W for word-optimized mapping.\n".
    "       * The default (time-optimized) mapping maps ref and sys metadata\n".
    "         so as to maximize the time overlap of mapped metadata events.\n".
    "       * Word-optimized mapping maps ref and sys metadata so as to\n".
    "         maximize the overlap in terms of the number of reference words\n".
    "         that are subsumed within the overlapping time interval.\n".
    "    -a <cfgs> Conditional analysis options for metadata detection performance:\n".
    "         c for performance versus channel,\n".
    "         f for performance versus file,\n".
    "         g for performance versus gender, and\n".
    "         s for performance versus speaker.\n".
    "    -A <cf> Conditional analysis options for word coverage performance:\n".
    "         c for performance versus channel,\n".
    "         f for performance versus file,\n".
    "    -t <time gap> The maximum time gap allowed between matching reference\n".
    "       and system output words (in seconds).  Default value is $default_tgap.\n".
    "    -T <time gap> The maximum time gap allowed between matching reference\n".
    "       and system output metadata (in seconds).  Default value is $default_Tgap.\n".
    "    -l <SU extent limit>  The maximum SU extent used to compute overlap\n".
    "       between reference and system output SU's.  For time-optimized SU\n".
    "       mapping this is the maximum time extent.  For word-optimized SU\n".
    "       mapping (using the -W option) this is the maximum number of words.\n".
    "       SU extent is limited to the last part of the SU. Default value is\n".
    "       $default_su_time_limit for time-optimized mapping, $default_su_word_limit for word-optimized mapping.\n".
    "    -u <uem-file> A file containing the evaluation partitions,\n".
    "       in UEM format.\n".
    "    -g <glm-file> A file containing word transformations used to\n".
    "       standardize the representation of words.\n".
    "    -o to include overlapping speech in MD evaluation.  With this option,\n".
    "       separate recognition passes are made for each reference speaker.\n".
    "    -c <collar> is the no-score zone around reference speaker segment\n".
    "       boundaries.  (Speaker Diarization output is not evaluated within\n".
    "       +/- collar seconds of a reference speaker segment boundary.)\n".
    "       Default value is $default_collar seconds.\n".
    "    -1 to limit scoring to those time regions in which only a single\n".
    "       speaker is speaking\n".
    "    -y <name> to select named no-eval conditions for metadata\n".
    "    -Y <name> to select named no-score conditions for metadata\n".
    "    -z <name> to select named no-eval conditions for speaker diarization\n".
    "    -Z <name> to select named no-score conditions for speaker diarization\n".
    "    -e to examine metadata mapping\n".
    "    -d to print word alignment and error calculation details\n".
    "    -D to print metadata event alignment and error calculation details\n".
    "    -m to print speaker mapping details for speaker diarization\n".
    "    -M FILE to print speaker mapping details for speaker diarization to a CSV file called 'FILE'\n".
    "    -v to print the event sequence for each diarization source file\n".
    "    -n to keep the directory names of the UEM source file entries\n".
    "OUTPUT:\n".
    "  Performance statistics are written to STDOUT.\n".
    "\n";

######
# Intro
my ($date, $time) = date_time_stamp();
print "command line (run on $date at $time) Version: $version  ", $0, " ", join(" ", @ARGV), "\n";

use vars qw ($opt_h $opt_w $opt_W $opt_d $opt_D $opt_R $opt_r $opt_S $opt_s $opt_l $opt_c $opt_x);
use vars qw ($opt_t $opt_T $opt_g $opt_p $opt_P $opt_o $opt_a $opt_A $opt_u $opt_1 $opt_m $opt_v $opt_e);
use vars qw ($opt_y $opt_Y $opt_z $opt_Z $opt_n $opt_M);
$opt_y = $opt_Y = $opt_z = $opt_Z = "DEFAULT";
use Getopt::Std;
getopts ('nhdDwWox1mvec:R:r:S:s:t:T:g:p:P:a:A:u:l:y:Y:z:Z:M:');
not defined $opt_h or die
    "\n$usage";
defined $opt_r or defined $opt_R or die
    "\nCOMMAND LINE ERROR:  no reference data specified$usage";
not defined $opt_r or not defined $opt_R or die
    "\nCOMMAND LINE ERROR:  both reference file list and reference file specified$usage";
defined $opt_s or defined $opt_S or die
    "\nCOMMAND LINE ERROR:  no system output data specified$usage";
not defined $opt_s or not defined $opt_S or die
    "\nCOMMAND LINE ERROR:  both system output file list and system output file specified$usage";
my $word_gap = defined $opt_t ? $opt_t : $default_tgap;
my $md_gap = $opt_W ? $default_Wgap : (defined $opt_T ? $opt_T : $default_Tgap);
my $su_extent_limit = defined $opt_l ? $opt_l :
    ($opt_W ? $default_su_word_limit : $default_su_time_limit);
$opt_W ? ($su_extent_limit >= 1 or die "\nCOMMAND LINE ERROR:  SU extent limit must be at least 1 for word-based MD alignment$usage") :
         ($su_extent_limit > 0 or die "\nCOMMAND LINE ERROR:  SU extent limit must be positive for time-based MD alignment$usage");
my $max_wd_delta_score = $opt_w ? $default_word_delta_score : $default_time_delta_score;
$max_wd_delta_score = $opt_p if defined $opt_p;
my $max_md_delta_score = $opt_W ? $default_word_delta_score : $default_time_delta_score;
$max_md_delta_score = $opt_P if defined $opt_P;
my $collar = defined($opt_c) ? $opt_c : $default_collar;
$collar >= 0 or die
    "\nCOMMAND LINE ERROR:  Speaker Diarization scoring collar ('$collar') must be non-negative$usage";
my $max_extend = $default_extend;
$opt_a = "" unless defined $opt_a;
$opt_A = "" unless defined $opt_A;
start_speaker_map_file($opt_M) if $opt_M;

my $noeval_md = eval_condition ($opt_y, $noeval_mds, "no-eval", "metadata");
my $noscore_md = eval_condition ($opt_Y, $noscore_mds, "no-score", "metadata");
my $noeval_sd = eval_condition ($opt_z, $noeval_sds, "no-score", "speaker diarization");
my $noscore_sd = eval_condition ($opt_Z, $noscore_sds, "no-score", "speaker diarization");

my %type_order = (NOSCORE        => 0,
		  NO_RT_METADATA => 1,
		  SEGMENT        => 2,
		  SPEAKER        => 3,
		  SU             => 4,
		  "A/P"          => 5,
		  "NON-SPEECH"   => 6,
		  EDIT           => 7,
		  FILLER         => 8,
		  IP             => 9,
		  CB             => 10,
		  "NON-LEX"      => 11,
		  LEXEME         => 12);
my %event_order = (END => 0,
		   MID => 1,
		   BEG => 2);
my %source_order = (REF => 0,
		    SYS => 1);

{
    my (%ref, %sys, $glm, $uem);

    print_parameters ();
    ($glm) = get_glm_data ($opt_g);
    get_rttm_file (\%ref, $opt_r, $glm);
    get_rttm_data (\%ref, $opt_R, $glm);
    get_rttm_file (\%sys, $opt_s, $glm);
    get_rttm_data (\%sys, $opt_S, $glm);

    ($uem) = get_uem_data ($opt_u, $opt_n);
    evaluate (\%ref, \%sys, $uem);
}

exit 0;

#################################

sub eval_condition {

    my ($name, $conditions, $exclusion, $evaluation) = @_;

    $name = "DEFAULT" unless $name;
    return $conditions->{$name} if defined $conditions->{$name};
    print STDERR "\nCOMMAND LINE ERROR:  unknown name ($name) of $exclusion conditions for $evaluation\n".
	"    available $exclusion conditions for $evaluation are:\n\n";
    foreach $name (sort keys %$conditions) {
	printf STDERR "%-24stype    subtype\n", "    for \"$name\":";
	foreach my $type (sort keys %{$conditions->{$name}}) {
	    foreach my $subt (sort keys %{$conditions->{$name}{$type}}) {
		printf STDERR "%28s    %s\n", $type, $subt;
	    }
	}
	print "\n";
    }
    die "$usage";
}    

#################################

sub print_parameters {

    print $opt_w ? "\nWord-based metadata alignment, max gap between matching words = $word_gap sec\n" :
	"\nTime-based metadata alignment\n";
    print "\nMetadata evaluation parameters:\n";
    $opt_W ? (print "    word-optimized metadata mapping\n".
	            "        max gap between matching metadata events = $md_gap words\n".
	            "        max extent to match for SU's = $su_extent_limit words\n") :
	     (print "    time-optimized metadata mapping\n".
		    "        max gap between matching metadata events = $md_gap sec\n".
		    "        max extent to match for SU's = $su_extent_limit sec\n");
    print "\nSpeaker Diarization evaluation parameters:\n".
	  "    The max time to extend no-score zones for NON-LEX exclusions is $max_extend sec\n".
	  "    The no-score collar at SPEAKER boundaries is $collar sec\n";
    printf "\nExclusion zones for evaluation and scoring are:\n".
	   "                             -----MetaData-----        -----SpkrData-----\n".
	   "     exclusion set name:%12s%11s%15s%11s\n".
	   "     token type/subtype      no-eval   no-score        no-eval   no-score\n",
	       $opt_y, $opt_Y, $opt_z, $opt_Z;
    print  "             (UEM)              X                         X\n";
    foreach my $type (sort keys %rttm_datatypes) {
	foreach my $subt (sort keys %{$rttm_datatypes{$type}}) {
	    next unless ($noeval_md->{$type}{$subt} or
			 $noscore_md->{$type}{$subt} or
			 $noeval_sd->{$type}{$subt} or
			 $noscore_sd->{$type}{$subt});
	    printf "%15s/%-14s", $type, $subt;
	    printf "%3s", $noeval_md->{$type}{$subt} ? "X" : "";
	    printf "%10s", $noscore_md->{$type}{$subt} ? "X" : "";
	    printf "%16s", $noeval_sd->{$type}{$subt} ? "X" : "";
	    printf "%10s\n", $noscore_sd->{$type}{$subt} ? "X" : "";
	}
    }
}

#################################

sub get_glm_data {

    my ($file) = @_;
    my ($record, @fields, $word, %words, %data);

    return unless defined $file;
    open DATA, $file or die
	"\nCOMMAND LINE ERROR:  unable to open glm file '$file'$usage";
    while ($record = <DATA>) {
	next if $record =~ /^\s*$/;
	next if $record =~ /^\s*(\[|\*|\%|\;)/;
        @fields = split /\s+=>\s+/, lc $record;
	shift @fields if $fields[0] eq "";
        next unless @fields > 1;
	$fields[0] =~ s/^\s+//;
        $fields[1] =~ s/[^a-z-'_ \.].*//;
        next if $fields[0] =~ /^\s*$/ or $fields[1] =~ /^\s*$/;
        $data{$fields[0]} = [split /\s+/, $fields[1]];
    }
    close DATA;
    return {%data};
}

#################################

sub get_uem_data {

    my ($file, $keepDirectoryPath) = @_;
    my ($record, @fields, $seg, $chnl, %data);

    return unless defined $file;
    open DATA, $file or die
	"\nCOMMAND LINE ERROR:  unable to open uem file '$file'$usage";
    while ($record = <DATA>) {
	next if $record =~ /^\s*[\#;]|^\s*$/;
	@fields = split /\s+/, $record;
	shift @fields if $fields[0] eq "";
	@fields >= 4 or die
	    ("\n\nFATAL ERROR:  insufficient number of fields in UEM record\n".
	     "    record is: '$record'\n\n");
	undef $seg;
	$seg->{FILE} = shift @fields;
	$seg->{CHNL} = lc shift @fields;
	$seg->{TBEG} = lc shift @fields;
	$seg->{TEND} = lc shift @fields;
        $seg->{FILE} =~ s/.*\/// if (! $keepDirectoryPath);      #strip directory 
        $seg->{FILE} =~ s/\.[^.]*//;   #strip file type
        $seg->{TBEG} =~ s/[^0-9\.]//g; #strip non-numeric (commas)
        $seg->{TEND} =~ s/[^0-9\.]//g; #strip non-numeric (commas)
	push @{$data{$seg->{FILE}}{$seg->{CHNL}}}, $seg;
    }
    close DATA;

#sort and check data
    foreach $file (keys %data) {
	foreach $chnl (keys %{$data{$file}}) {
	    @{$data{$file}{$chnl}} =
		sort {$a->{TBEG} <=> $b->{TBEG}} @{$data{$file}{$chnl}};
	    my $prev_seg;
	    foreach $seg (@{$data{$file}{$chnl}}) {
		$seg->{TEND} > $seg->{TBEG} or die
		    "\n\nFATAL ERROR:  non-positive evaluation segment length in UEM data for file $file, channel $chnl\n\n";
		not defined $prev_seg or $seg->{TBEG} >= $prev_seg->{TEND} or die
		    ("\n\nFATAL ERROR:  UEM file has overlapping evaluation segments\n".
		     "    file $file, channel $chnl:  ($prev_seg->{TBEG},$prev_seg->{TEND}),".
		     " ($seg->{TBEG},$seg->{TEND})\n\n");
		$prev_seg = $seg;
	    }
	}
    }
    return {%data};
}

#################################

sub get_rttm_data {

    my ($data, $list, $glm) = @_;

    return unless defined $list;
    open LIST, $list or die
	"\nCOMMAND LINE ERROR:  unable to open file list '$list'$usage";
    while (my $file = <LIST>) {
	get_rttm_file ($data, $file, $glm);
    }
    close LIST;
}    

#################################

sub get_rttm_file {

    my ($data, $rttm_file, $glm) = @_;
    my ($record, @fields, $data_type, $file, $chnl, $word, @words, $token);

    return unless defined $rttm_file;
    open DATA, $rttm_file or die
	"\nCOMMAND LINE ERROR:  unable to open RTTM file '$rttm_file'$usage";
    while ($record = <DATA>) {
	next if $record =~ /^\s*[\#;]|^\s*$/;
	@fields = split /\s+/, $record;
	shift @fields if $fields[0] eq "";
	@fields >= 9 or die
	    ("\n\nFATAL ERROR:  insufficient number of fields in RTTM file '$rttm_file'\n".
	     "    input RTTM record is: '$record'\n\n");
	$data_type = uc shift @fields;
	undef $token;
	$token->{TYPE} = $data_type;
	$token->{FILE} = $file = shift @fields;
	$token->{CHNL} = $chnl = lc shift @fields;
	$token->{TBEG} = lc shift @fields;
	$token->{TBEG} =~ s/\*//;
	$token->{TDUR} = lc shift @fields;
	$token->{TDUR} =~ s/\*//;
	$token->{TDUR} = 0 if $token->{TDUR} eq "<na>";
	$token->{TDUR} >= 0 or die
	    ("\n\nFATAL ERROR -- negative metadata duration in file $file,'\n".
	     "    input RTTM record is: '$record'\n\n");
	$token->{WORD} = lc shift @fields;
	$token->{SUBT} = lc shift @fields;
	$rttm_datatypes{$token->{TYPE}}{$token->{SUBT}} or die
	    ("\n\nFATAL ERROR:  unknown RTTM data type/subtype ('$token->{TYPE}'/'$token->{SUBT}') in file $rttm_file\n".
	     "    input RTTM record is: '$record'\n\n");
	$token->{SPKR} = shift @fields;
	$token->{CONF} = lc shift @fields;
	$token->{CONF} = "-"       unless defined $token->{CONF};
	$token->{SPKR} = "<na>" unless defined $token->{SPKR};
	if ($data_type eq "SPKR-INFO") {
	    not defined $data->{$file}{$chnl}{$data_type}{$token->{SPKR}} or die
		("\n\nFATAL ERROR:  multiple $data_type records for speaker $token->{SPKR} in file $file\n".
		 "    input RTTM record is: '$record'\n\n");
	    defined $spkr_subtypes{$token->{SUBT}} or die
		("\n\nFATAL ERROR:  unknown $data_type subtype ($token->{SUBT}) in file '$file'\n".
		 "    input RTTM record is: '$record'\n\n");
	    $data->{$file}{$chnl}{$data_type}{$token->{SPKR}}{GENDER} = $token->{SUBT};
	}
	else {
	    $token->{TEND} = $token->{TBEG}+$token->{TDUR};
	    $token->{TMID} = $token->{TBEG}+$token->{TDUR}/2;
	}

	if ($data_type eq "LEXEME") {
	    $token->{WTYP} = ($token->{SUBT} =~ /^fp$/         ? "fp"      :
			      ($token->{SUBT} =~ /^frag$/      ? "frag"    :
			       ($token->{SUBT} =~ /^un-lex$/   ? "un-lex"  :
				($token->{SUBT} =~ /^for-lex$/ ? "for-lex" : "lex"))));
	    @words = standardize_word ($token, $glm);
	    foreach $word (@words) {
		push @{$data->{$file}{$chnl}{LEXEME}}, $word;
		push @{$data->{$file}{$chnl}{RTTM}}, $word;
	    }
	}
	elsif ($data_type eq "SPEAKER") {
	    push @{$data->{$file}{$chnl}{SPEAKER}{$token->{SPKR}}}, $token;
	    push @{$data->{$file}{$chnl}{RTTM}}, $token;
	}
	elsif ($md_subtypes{$token->{TYPE}}) {
	    defined $md_subtypes{$token->{TYPE}}{$token->{SUBT}} or die
		("\n\nFATAL ERROR:  unknown $data_type subtype ($token->{SUBT}) in file '$file'\n".
		 "    input RTTM record is: '$record'\n\n");
	    push @{$data->{$file}{$chnl}{$data_type}}, $token;
	    push @{$data->{$file}{$chnl}{RTTM}}, $token;
	}
	elsif ($data_type ne "SPKR-INFO") {
	    push @{$data->{$file}{$chnl}{RTTM}}, $token;
	}
    }
    close DATA;

#sort and check data
    foreach $file (keys %$data) {
	foreach $chnl (keys %{$data->{$file}}) {
	    foreach $data_type (keys %{$data->{$file}{$chnl}}) {
		next if $data_type eq "SPKR-INFO";
		if ($data_type eq "SPEAKER") {
		    foreach my $spkr (keys %{$data->{$file}{$chnl}{$data_type}}) {
			my $gender = $data->{$file}{$chnl}{"SPKR-INFO"}{$spkr}{GENDER};
			$gender = $data->{$file}{$chnl}{"SPKR-INFO"}{$spkr}{GENDER} = "unknown" if not $gender;
			@{$data->{$file}{$chnl}{$data_type}{$spkr}} =
			    sort {$a->{TMID}<=>$b->{TMID}} @{$data->{$file}{$chnl}{$data_type}{$spkr}};
			my $prev_token;
			foreach $token (@{$data->{$file}{$chnl}{$data_type}{$spkr}}) {
			    $token->{SUBT} = $gender;
			    next unless $prev_token;
			    not $prev_token or $token->{TBEG} >= $prev_token->{TEND}-$epsilon or die
				("\n\nFATAL ERROR:  RTTM file has overlapping $data_type tokens for speaker $spkr\n".
				 "    in file $file, channel $chnl:  ($prev_token->{TBEG},$prev_token->{TEND}),".
				 " ($token->{TBEG},$token->{TEND})\n\n");
			    $prev_token = $token;
			}
		    }
		}
		else {
		    @{$data->{$file}{$chnl}{$data_type}} =
			sort {$a->{TMID} <=> $b->{TMID}} @{$data->{$file}{$chnl}{$data_type}};
		}
	    }
	}
    }
}

#################################

sub evaluate {

    my ($ref_data, $sys_data, $uem_data) = @_;
    my ($uem, $uem_sd_eval, $uem_sd_score, $uem_md_eval, $uem_md_score);
    my ($ref_wds, $sys_wds, $ref_mds, $sys_mds, $type, %scores, $ref_rttm, $sys_rttm);

    foreach my $file (sort keys %$ref_data) {
	foreach my $chnl (sort keys %{$ref_data->{$file}}) {
	    $ref_rttm = $ref_data->{$file}{$chnl}{RTTM};
	    $sys_rttm = $sys_data->{$file}{$chnl}{RTTM};
	    $ref_wds = $ref_data->{$file}{$chnl}{LEXEME} ? $ref_data->{$file}{$chnl}{LEXEME} : [];
	    $sys_wds = $sys_data->{$file}{$chnl}{LEXEME} ? $sys_data->{$file}{$chnl}{LEXEME} : [];
	    $uem = $uem_data->{$file}{$chnl};
	    $uem = uem_from_rttm ($ref_rttm) if not defined $uem;
	    @$ref_wds > 0 or not $opt_w or die
		"\n\nFATAL ERROR:  no reference words for file '$file' and channel '$chnl'\n\n";
	    @$sys_wds > 0 or not $opt_w or die
		"\n\nFATAL ERROR:  no system output words for file '$file' and channel '$chnl'\n".
		    "              Words are required for word-mediated alignment\n\n";
	    if ($ref_wds and ($opt_w or $opt_e)) {
	        tag_words_with_metadata_attributes ($ref_rttm, $ref_wds);
	        tag_words_with_metadata_attributes ($sys_rttm, $sys_wds);
	        perform_word_alignment ($file, $chnl, $ref_wds, $sys_wds, $uem);
	    }
	    $uem_md_eval = add_exclusion_zones_to_uem ($noeval_md, $uem, $ref_rttm);
	    $uem_md_score = add_exclusion_zones_to_uem ($noscore_md, $uem_md_eval, $ref_rttm);
	    $uem_md_score = exclude_overlapping_speech_from_uem ($uem_md_score, $ref_rttm) if $opt_1;
 	    tag_scoreable_words ($ref_wds, $uem_md_score);
	    foreach $type (sort keys %md_subtypes) {
		$ref_mds = $ref_data->{$file}{$chnl}{$type};
		next unless defined $ref_mds;
		@$ref_wds > 0 or die
		    "\n\nFATAL ERROR:  no reference words for file '$file' and channel '$chnl'\n\n";
		$sys_mds = $sys_data->{$file}{$chnl}{$type};
		$sys_mds = $sys_data->{$file}{$chnl}{$type} = [] unless defined $sys_mds;
		map_metadata_to_words ($sys_mds, $sys_wds, $ref_mds, $ref_wds);
		discard_unevaluated_metadata ($uem_md_eval, $type, $ref_mds, $ref_wds, "REF");
		next if @$ref_mds == 0;
		align_data ($ref_mds, $sys_mds, "", \&md_score, $max_md_delta_score);
		trace_best_path ($ref_mds, $sys_mds);
		discard_metadata_subtype ("EDIT", "complex", $ref_mds, $sys_mds) if $type eq "EDIT" and $opt_x;
		discard_metadata_subtype ("SU", "unannotated", $ref_mds, $sys_mds) if $type eq "SU";
		discard_unevaluated_metadata ($uem_md_eval, $type, $sys_mds, $ref_wds, "SYS");
		($scores{$type}{$file}{$chnl}) = score_metadata_path
		    ($type, $file, $chnl, $ref_mds, $sys_mds, $ref_wds);
	    }

	    $ref_mds = $ref_data->{$file}{$chnl}{SPEAKER};
	    if (defined $ref_mds) {
		@$ref_wds > 0 or not $opt_W or die
		    "\n\nFATAL ERROR:  no reference words for file '$file' and channel '$chnl'\n\n";
		$uem_sd_eval = add_exclusion_zones_to_uem ($noeval_sd, $uem, $ref_rttm);
		$sys_mds = $sys_data->{$file}{$chnl}{SPEAKER};
		$sys_mds = $sys_data->{$file}{$chnl}{SPEAKER} = {} unless defined $sys_mds;
		map_spkrdata_to_words ($sys_mds, $sys_wds, $ref_mds, $ref_wds);
		($scores{SPEAKER}{$file}{$chnl}) = score_speaker_diarization
		    ($file, $chnl, $ref_mds, $sys_mds, $ref_wds, $uem_sd_eval, $ref_rttm);
	    }

	    if ($opt_e) {
		discard_unevaluated_metadata ($uem, "LEXEME", $ref_rttm);
		discard_unevaluated_metadata ($uem, "LEXEME", $sys_rttm);
		discard_unevaluated_metadata ($uem_md_eval, "", $ref_rttm);
		discard_metadata_subtype ("EDIT", "complex", $ref_rttm, $sys_rttm) if $opt_x;
		discard_metadata_subtype ("SU", "unannotated", $ref_rttm, $sys_rttm);
		discard_unevaluated_metadata ($uem_md_eval, "", $sys_rttm);
		display_metadata_mapping ($file, $chnl, $ref_rttm, $sys_rttm, $ref_wds);
	    }
	}
    }

    foreach $type (sort keys %md_subtypes) {
	md_performance_analysis ($type, $scores{$type}, $md_subtypes{$type}, $ref_data)
	    if $scores{$type};
    }
    sd_performance_analysis ($scores{SPEAKER}, \%spkr_subtypes)
	if $scores{SPEAKER};
}

#################################

sub perform_word_alignment {

    my ($file, $chnl, $ref_wds, $sys_wds, $uem) = @_;

    my @ref_wds = @$ref_wds;
    my @sys_wds = @$sys_wds;
    discard_unevaluated_words ($uem, \@ref_wds);
    discard_unevaluated_words ($uem, \@sys_wds);
    @ref_wds > 0 or die
	"\n\nFATAL ERROR:  no reference words in UEM portion of file '$file' and channel '$chnl'\n\n";
    @sys_wds > 0 or not $opt_w or die
	"\n\nFATAL ERROR:  no system output words in UEM portion of file '$file' and channel '$chnl'\n".
	    "              Words are required for word-mediated alignment\n\n";
    return unless @sys_wds > 0;

    if ($opt_o) {
	foreach my $spkr (word_kinds ($ref_wds, "SPKR")) {
	    align_data ($ref_wds, $sys_wds, $spkr, \&word_score, $max_wd_delta_score);
	    trace_best_path ($ref_wds, $sys_wds, $spkr);
	}
	decide_who_spoke_the_words ($ref_wds, $sys_wds);
    }
    else {
	align_data ($ref_wds, $sys_wds, "", \&word_score, $max_wd_delta_score);
	trace_best_path ($ref_wds, $sys_wds);
    }

#map system output word times to ref words
    foreach my $wd (@$sys_wds) {
	$wd->{RTBEG} = adjust_sys_time_to_ref ($wd->{TBEG}, $sys_wds);
	$wd->{RTEND} = adjust_sys_time_to_ref ($wd->{TEND}, $sys_wds);
	$wd->{RTDUR} = $wd->{RTEND} - $wd->{RTBEG};
	$wd->{RTMID} = $wd->{RTBEG} + $wd->{RTDUR}/2;
    }
    score_word_path ($file, $chnl, $ref_wds, $sys_wds) if $opt_d;
}

################################

sub time_in_eval_partition {

    my ($time, $uem_eval) = @_;

    return 1 unless defined $uem_eval; #not using UEM partition specification
    foreach my $partition (@$uem_eval) {
	return 1 if event_covers_time ($partition, $time);
    }
    return 0;
}

#################################

sub discard_unevaluated_words {

    my ($uem, $wds) = @_;

    for (my $index=0; $index<@$wds; $index++) {
	splice (@$wds, $index--, 1)
	    if ($wds->[$index]{TYPE} eq "LEXEME" and
		not time_in_eval_partition ($wds->[$index]{TMID}, $uem));
    }
}

#################################

sub discard_unevaluated_metadata {

    my ($uem_eval, $type, $mds, $ref_wds, $src) = @_;

    for (my $index=0; $index<@$mds; $index++) {
	my $md = $mds->[$index];
	next if (($type and $md->{TYPE} ne $type) or
		 (not $type and not $md_subtypes{$md->{TYPE}}) or
		 $md->{MAPPTR} or
		 md_in_uem ($md, $uem_eval));
	warn_if_discarded_md_covers_scored_lexemes ($md, $ref_wds, $uem_eval, $src) if $ref_wds;
	splice (@$mds, $index--, 1);
    }
}

#################################

sub warn_if_discarded_md_covers_scored_lexemes {

    my ($md, $ref_wds, $uem, $source) = @_;
    my ($wbeg, $wend, $index);
    
    ($wbeg, $wend) = md_word_indices ($md, $ref_wds);

    for ($index=$wbeg; $index<=$wend; $index++) {
	next unless ($ref_wds->[$index]{SCOREABLE} and
		     time_in_eval_partition ($ref_wds->[$index]{TMID}, $uem));
	warn "\nWARNING:  A $source metadata event is being deleted that covers evaluated reference LEXEMEs\n".
	    "    (type=$md->{TYPE}, subtype=$md->{SUBT}, spkr=$md->{SPKR}, TBEG=$md->{TBEG}, TEND=$md->{TEND})\n";
	last;
    }	
}

#################################

sub discard_metadata_subtype {

    my ($type, $subtype, $ref_mds, $sys_mds) = @_;
    my ($iref, $isys, $ref_md, $sys_md);

#discard all sys $type events that map to a ref event with subtype = $subtype
#or that are unmapped and have midpoints that lie within a ref event with subtype = $subtype
    for ($iref=0; $iref<@$ref_mds; $iref++) {
	$ref_md = $ref_mds->[$iref];
	next unless ($ref_md->{TYPE} eq $type and
		     $ref_md->{SUBT} eq $subtype);
	for ($isys=0; $isys<@$sys_mds; $isys++) {
	    $sys_md = $sys_mds->[$isys];
	    splice (@$sys_mds, $isys--, 1)
		if ($sys_md->{TYPE} eq $type and
		    (($sys_md->{MAPPTR} and $sys_md->{MAPPTR}{SUBT} eq $subtype) or
		     (not $sys_md->{MAPPTR} and event_covers_time ($ref_md, $sys_md->{RTMID}))));
	}

#discard all ref $type/$subtype events
	splice (@$ref_mds, $iref--, 1);
    }
}

#################################

sub tag_scoreable_words {

    my ($wds, $uem_eval) = @_;

    foreach my $wd (@$wds) {
	$wd->{SCOREABLE} = time_in_eval_partition ($wd->{TMID}, $uem_eval);
    }
}

#################################

sub tag_words_with_metadata_attributes {

    my ($mds, $wds) = @_;
    my ($md, $iwbeg, $iwend, $iw, $wd, $type);

    foreach $md (@$mds) {
	$type = $md->{TYPE};
	next unless $type =~ /^(FILLER|EDIT|SU|IP)$/;
	($iwbeg, $iwend) = md_word_indices ($md, $wds);
	if ($type =~ /^(FILLER|EDIT)$/) {
	    for ($iw=$iwbeg; $iw<=$iwend; $iw++) {
		$wds->[$iw]{ATTRIBUTES}{$md->{TYPE}} = $md->{SUBT};
	    }
	}
	elsif ($type =~ /^(SU|IP)$/) {
	    $wds->[$iwend]{ATTRIBUTES}{$md->{TYPE}} = $md->{SUBT};
	}
    }
    return;
}

#################################

sub tag_ref_words_with_metadata_info {

    my ($mds, $wds, $src) = @_;
    my ($md, $iwbeg, $iwend, $iw, $type);

    foreach $md (@$mds) {
	$type = $md->{TYPE};
	($iwbeg, $iwend) = $src eq "REF" ?
	    ($md->{WBEG}, $md->{WEND}) : ($md->{RWBEG}, $md->{RWEND}) ;
	if ($type =~ /^(FILLER|EDIT)$/) {
	    for ($iw=max($iwbeg,0); $iw<=min($iwend,@$wds-1); $iw++) {
		$wds->[$iw]{"$src-$type"}{$md->{SUBT}}{MAP}++;
	    }
	}
	elsif ($type =~ /^(SU|IP)$/) {
	    $iwend = min(max($iwend,0),@$wds-1);
	    $wds->[$iwend]{"$src-$type"}{$md->{SUBT}}{defined $md->{MAPPTR} ? "MAP" : "NOT"}++;
	}
    }
    return;
}

#################################

sub md_performance_analysis {

    my ($metadata_type, $counts, $subtypes, $ref_data) = @_;
    my ($file, $chnl, $spkr, $word, $type, $type_counts, $key);
    my (@files, @chnls, @spkrs, @types, %nevent, %nwerr);
    my ($subtype, $sys_subtype, %nconf, %offsets);

#compute marginal counts
    @files = keys %$counts;
    foreach $file (@files) {
	@chnls = keys %{$counts->{$file}};
	foreach $chnl (@chnls) {
	    $type_counts = $counts->{$file}{$chnl};
	    foreach $type ("REF", "DEL", "INS", "SUB") {
	        next unless defined $type_counts->{WORDS}{$type};
		$nwerr{ALL}{$type} += $type_counts->{WORDS}{$type};
		$nwerr{"c=$chnl f=$file"}{$type} += $type_counts->{WORDS}{$type} if $opt_A =~ /c/i and $opt_A =~ /f/i;
		$nwerr{"c=$chnl"}{$type} += $type_counts->{WORDS}{$type} if $opt_A =~ /c/i and not $opt_A =~ /f/i;
		$nwerr{"f=$file"}{$type} += $type_counts->{WORDS}{$type} if $opt_A =~ /f/i and not $opt_A =~ /c/i;
	    }
	    foreach $type ("WBEG", "WEND") {
		foreach $key (keys %{$type_counts->{WORD_OFFSET}{$type}}) {
		    $offsets{ALL}{$type}{$key} += $type_counts->{WORD_OFFSET}{$type}{$key};
		}
	    }
	    my $spkr_info = $ref_data->{$file}{$chnl}{"SPKR-INFO"};
	    $spkr_info->{unknown}{GENDER} = "unknown" unless defined $spkr_info->{unknown};
	    foreach $type (keys %$type_counts) {
		next unless $type =~ /^(REF|DEL|INS|SUB|CONFUSION)$/;
		@spkrs = keys %{$type_counts->{$type}};
		foreach $spkr (@spkrs) {
		    my $gndr = $spkr_info->{$spkr}{GENDER};
		    foreach $subtype (keys %$subtypes) {
			my $count = $type_counts->{$type}{$spkr}{$subtype};
			next unless $count;
			if ($type eq "CONFUSION") {
			    foreach $sys_subtype (keys%$count) {
				$nconf{ALL}{$subtype}{$sys_subtype} += $count->{$sys_subtype};
				$nconf{ALL}{$subtype}{$sys_subtype} = 0 if not $nconf{ALL}{$subtype}{$sys_subtype};
				$nconf{ALL}{$sys_subtype}{$subtype} = 0 if not $nconf{ALL}{$sys_subtype}{$subtype};
			    }
			    next;
			}
			$nconf{ALL}{$subtype}{"{Miss}"} += $count if $type eq "DEL";
			$nconf{ALL}{"{FA}"}{$subtype} += $count if $type eq "INS";
			$nconf{ALL}{$subtype}{"{Miss}"} = 0 unless defined $nconf{ALL}{$subtype}{"{Miss}"};
			$nconf{ALL}{"{FA}"}{$subtype} = 0 unless defined $nconf{ALL}{"{FA}"}{$subtype};
			$nevent{ALL}{$type} += $count;
			$nevent{"c=$chnl f=$file"}{$type} += $count if $opt_a =~ /c/i and $opt_a =~ /f/i;
			$nevent{"c=$chnl"}{$type} += $count if $opt_a =~ /c/i and not $opt_a =~ /f/i;
			$nevent{"f=$file"}{$type} += $count if $opt_a =~ /f/i and not $opt_a =~ /c/i;
			$nevent{"s=$spkr"}{$type} += $count if $opt_a =~ /s/i;
			$nevent{"g=$gndr"}{$type} += $count if $opt_a =~ /g/i;
		    }
		}
	    }
	}
    }
    print_md_scores ($metadata_type, \%nevent, \%nconf, \%offsets, \%nwerr);
}

#################################

sub print_offset_stats {

    my ($counts) = @_;
    my (@offsets, $count, $min, $max, $i);

    @offsets = (keys %{$counts->{WBEG}}, keys %{$counts->{WEND}});
    $min = min (-3, @offsets);
    $max = max (3, @offsets);
    print "  word offsets:  <-3  ";
    for ($i=-3; $i<=3; $i++) {
	printf "%5d", $i;
    }
    print "     >3\n";
    print "           BEG:";
    for ($count=0,$i=$min; $i<-3; $i++) {
	$count += $counts->{WBEG}{$i} if defined $counts->{WBEG}{$i};
    }	    
    printf "%5d  ", $count if defined $count;
    print "    -  ", unless defined $count;
    for ($i=-3; $i<=3; $i++) {
	$count = $counts->{WBEG}{$i};
	printf "%5d", $count if defined $count;
	print "    -", unless defined $count;
    }
    for ($count=0,$i=4; $i<=$max; $i++) {
	$count += $counts->{WBEG}{$i} if defined $counts->{WBEG}{$i};
    }	    
    printf "%7d", $count if defined $count;
    print "      -", unless defined $count;
    
    print "\n           END:";
    for ($count=0,$i=$min; $i<-3; $i++) {
	$count += $counts->{WEND}{$i} if defined $counts->{WEND}{$i};
    }	    
    printf "%5d  ", $count if defined $count;
    print "    -  ", unless defined $count;
    for ($i=-3; $i<=3; $i++) {
	$count = $counts->{WEND}{$i};
	printf "%5d", $count if defined $count;
	print "    -", unless defined $count;
    }
    for ($count=0,$i=4; $i<=$max; $i++) {
	$count += $counts->{WEND}{$i} if defined $counts->{WEND}{$i};
    }	    
    printf "%7d", $count if defined $count;
    print "      -", unless defined $count;
    print "\n";
}

#################################

sub print_md_scores {

    my ($metadata_type, $event_counts, $conf_counts, $offset_counts, $word_counts) = @_;
    my ($type, $nerr, $norm, $name, $ref, $sys, $category, $counts);
    my ($count, $min, $max, $i, @offsets);
    my $head_format = "%36s   %5s %5s %5s   %6s %6s %6s   %6s %6s\n";
    my $data_format = "%-28.28s   %5d   %5d %5d %5s   %6.2f %6.2f %6.2f   %6.2f %6.2f\n";
    my @header = ("Nref", "Ndel", "Nins", "Nsub", "%Del", "%Ins", "%Sub", "%D+I", "%Tot");

    $counts = $word_counts->{ALL};
    $nerr = $counts->{DEL} + $counts->{INS};
    $nerr += $counts->{SUB} if $metadata_type =~ /^(SU|FILLER)$/;
    printf "\n*** Performance analysis for %ss ***  overall error SCORE = %.2f%s\n",
        $metadata_type,	100*$nerr/max($counts->{REF},$epsilon), "%";

#metadata word detection
    print "\nSU (exact) end detection statistics" if $metadata_type eq "SU";
    print "\nIP (exact) detection statistics" if $metadata_type eq "IP";
    print "\n$metadata_type word coverage statistics" unless $metadata_type =~ /^(SU|IP)$/;
    print " -- in terms of reference words\n";
    printf $head_format, @header;
    foreach $category (sort keys %$word_counts) {
	printf $data_format, ($category ne "ALL" ? $category : " "x17 ."ALL",
			      error_output ($word_counts->{$category}));
    }

#metadata event detection
    print "\n$metadata_type detection statistics -- in terms of \# of $metadata_type"."s\n";
    printf $head_format, @header;
    foreach $category (sort keys %$event_counts) {
	printf $data_format, ($category ne "ALL" ? $category : " "x17 ."ALL",
			      error_output ($event_counts->{$category}));
    }

#metadata event classification
    print "\n$metadata_type detection confusion matrix -- in terms of \# of $metadata_type"."s\n";
    foreach $category (sort keys %$conf_counts) {
	$counts = $conf_counts->{$category};
	printf "%24.24s", "$category - ref\\sys";
	foreach $name (sort keys %$counts, "{Miss}") {
	    next if $name eq "{FA}";
	    print "    " if $name eq "{Miss}";
	    printf "%10.8s", $name;
	}
	print "\n";
	foreach $ref (sort keys %$counts) {
	    print "\n" if $ref eq "{FA}";
	    printf "%24.24s", $ref;
	    foreach $sys (sort keys %$counts, "{Miss}") {
		next if $sys eq "{FA}" or ($ref eq "{FA}" and $sys eq "{Miss}");
		print "    " if $sys eq "{Miss}";
		printf "%8d  ", $counts->{$ref}{$sys} ? $counts->{$ref}{$sys} : 0;
	    }
	    print "\n";
	}
    }

#offsets
    foreach $category (sort keys %$offset_counts) {
	print "\n$metadata_type word offset statistics for $category data\n";
	print_offset_stats ($offset_counts->{$category});
    }
}

#################################

sub error_output {

    my ($counts) = @_;
    my (@output, $item, $nerr);

    foreach $item ("REF", "DEL", "INS", "SUB") {
	$counts->{$item} = 0 unless defined $counts->{$item};
	push @output, $counts->{$item};
	$nerr += $counts->{$item} unless $item eq "REF";
    }

    my $norm = 100/max($counts->{REF},$epsilon);
    foreach my $item ("DEL", "INS", "SUB") {
	push @output, min(999.99,$norm*$counts->{$item});
    }
    my $dpi = $counts->{"DEL"}+$counts->{"INS"};
    my $tot = $dpi+$counts->{"SUB"};
    push @output, min(999.99,$norm*$dpi), min(999.99,$norm*$tot);
    return @output;
}

#################################

sub word_kinds {

    my ($words, $kind) = @_;
    my ($word, %count);

    foreach $word (@$words) {
	$count{$word->{$kind}}++;
    }
    return sort keys %count;
}

#################################

sub standardize_word {

    my ($word, $glm) = @_;
    my (@split_word, @words, $tbeg, $tdur, $part, $new_word);

    $word->{WORD} =~ lc $word->{WORD}; #lower case

    if (defined $glm->{$word->{WORD}}) { #split glm words
	@split_word = @{$glm->{$word->{WORD}}};
    }
    elsif ($word->{WORD} =~ /^([^-]+|mm-hmm|uh-huh|um-hmm)$/) {
	return $word;
    }
    elsif ($word->{WORD} =~ /.+-.+/) { #split hyphenated words
	$word->{WORD} =~ s/(.+)-(.+)/$1 $2/g;
	@split_word = split /\s+/, $word->{WORD};
    }
    else { #don't split word
	return $word;
    }

#split word and prorate time equally to each part
    $tbeg = $word->{TBEG};
    $tdur = $word->{TDUR}/@split_word;
    foreach $part (@split_word) {
	$new_word = {FILE => $word->{FILE}, CHNL => $word->{CHNL}, TBEG => $tbeg,
		     TDUR => $tdur, TEND => $tbeg+$tdur, TMID => $tbeg+$tdur/2,
		     WORD => $part, CONF => $word->{CONF}, SPKR => $word->{SPKR},
		     TYPE => $word->{TYPE}, SUBT => $word->{SUBT}, WTYP => $word->{WTYP}};
	push @words, $new_word;
	$tbeg += $tdur;
    }
    return @words;
}

#################################

sub decide_who_spoke_the_words {

    my ($ref_wds, $sys_wds) = @_;
    my ($ref_index, $ref_word, $sys_index, $index, $word, $md_index, $md);
    my ($sys_word, $spkr, $score, $best_spkr, $best_score, @speakers);

#select the best ref word for each STT output word that has multiple reference word matches
    for ($sys_index=0; $sys_index<@$sys_wds; $sys_index++) {
	$sys_word = $sys_wds->[$sys_index];
	next unless defined $sys_word->{SPKRS};
	undef $best_score;
	@speakers = sort keys %{$sys_word->{SPKRS}};
	next unless @speakers > 1;
	foreach $spkr (@speakers) {
	    next unless defined $sys_word->{SPKRS}{$spkr};
	    $ref_word = $sys_word->{SPKRS}{$spkr}{REFPTR};
	    $score = $ref_word->{PATHS}{$sys_index}{SCORE};
	    next if defined $best_score and $best_score > $score;
	    $best_score = $score;
	    $best_spkr = $spkr;
	}
	next unless defined $best_score;
	foreach $spkr (@speakers) {
	    next if $spkr eq $best_spkr;
	    $sys_word->{SPKRS}{$spkr} = undef;
	    $ref_word = $sys_word->{SPKRS}{$best_spkr}{REFPTR};
	}
    }
}

#################################

sub event_covers_time {

    my ($event, $time) = @_;

    return ($time < $event->{TBEG} or
	    $time > $event->{TEND}) ? 0 : 1;
}

#################################

sub word_score {

    my ($ref_word, $sys_word) = @_;
    my ($tbeg, $tend, $rw, $sw, $score, $word);
    my ($attribute, $attributes, $ref_attributes, $sys_attributes);

#compute joint word coverage
    $score = 0;
    if (defined $ref_word and defined $sys_word) {
	return undef unless overlap ($ref_word, $sys_word, $word_gap);
	if (($ref_attributes = $ref_word->{ATTRIBUTES}) and
	    ($sys_attributes = $sys_word->{ATTRIBUTES})) {
	    foreach $attribute ("EDIT", "FILLER", "IP", "SU") {
	        next unless (defined $ref_attributes->{$attribute} and
			     defined $sys_attributes->{$attribute});
		$score += ($ref_attributes->{$attribute} eq
			   $sys_attributes->{$attribute}) ? 0.02 : 0.01;
	    }
	}
	return $score if #both word type and word spelling match
	    ((   $ref_word->{WORD} eq $sys_word->{WORD} and
		 $ref_word->{WTYP} eq $sys_word->{WTYP})
	     
	     or ($ref_word->{WTYP} eq "lex" and 
		 $sys_word->{WTYP} eq "frag" and
		 ($sw = $sys_word->{WORD}, $sw =~ s/^-*|-*$//g, $sw) #make sure that $sw is non-null
		 and ($ref_word->{WORD} =~ /$sw/))
	     
	     or ($ref_word->{WTYP} eq "frag" and 
		 $sys_word->{WTYP} eq "lex" and
		 ($rw = $ref_word->{WORD}, $rw =~ s/^-*|-*$//g, $rw) #make sure that $rw is non-null
		 and ($sys_word->{WORD} =~ /$rw/))
	     
	     or ($ref_word->{WTYP} eq "fp" and 
		 $sys_word->{WTYP} eq "fp")
	     
	     or ($ref_word->{WTYP} eq "frag" and 
		 $sys_word->{WTYP} eq "frag"));
	
	return $score - 0.1*max(1,ref_count($ref_word)) if #word type match, except for lex's
	    ((   $ref_word->{WTYP} eq $sys_word->{WTYP} and
		 $ref_word->{WTYP} ne "lex"));
	
	return $score - max(1,ref_count($ref_word),ref_count($sys_word));
    }
    $word = defined $ref_word ? $ref_word : $sys_word;
    return 0 unless defined $word;
    $score = $word->{WTYP} eq "lex" ? -ref_count($word) : -0.2*max(1,ref_count($word));
    $attributes = $word->{ATTRIBUTES};
    if (defined $attributes) {
	foreach $attribute ("EDIT", "FILLER", "IP", "SU") {
	    $score += 0.005 if defined $word->{$attribute};
	}
    }
    return $score;
}

#################################

sub wd_err_count {

    my ($ref_word, $sys_word) = @_;
    
    my $word_score = word_score($ref_word,$sys_word);
    return (defined $word_score and $word_score > -0.5) ? 0 : 1;
}

#################################

sub ref_count {

    my ($word) = @_;

    return 0 unless defined $word;
    return 0 if $word->{WTYP} =~ /^(non-lex|misc)$/;

#hyphenated words get a count of 2 (except for mm-hmm, uh-huh and hm-hmm)
    my $WORD = $word->{WORD};
    $WORD =~ s/^-*|-*$//g;
    return $WORD =~ /^([^-]+|mm-hmm|uh-huh|um-hmm)$/ ? 1 : 2;
}

#################################

sub overlap {

    my ($ref, $sys, $tgap) = @_;
    
    return 0 unless $ref and $sys;
    $tgap = 0 unless defined $tgap;
    my $tovl = (min($ref->{TEND}, $sys->{TEND}) -
		max($ref->{TBEG}, $sys->{TBEG})) + $tgap;
    return $tovl > 0 ? $tovl/(1 + $tgap/max($ref->{TDUR},$epsilon)) : 0;
}

################################

sub md_in_uem {

    my ($md, $uem_eval) = @_;

    return 1 unless defined $uem_eval; #not using UEM partition specification
    foreach my $partition (@$uem_eval) {
        return 1 if ($md->{TEND} <= $partition->{TEND}+$epsilon and
		     $md->{TBEG} >= $partition->{TBEG}-$epsilon);
    }
    return 0;
}

#################################

sub map_spkrdata_to_words {

    my ($sys_mds, $sys_wds, $ref_mds, $ref_wds) = @_;
    my ($spkr, $md, @ref_spkr_mds, @sys_spkr_mds);

    foreach $spkr (keys %$ref_mds) {
	foreach $md (@{$ref_mds->{$spkr}}) {
	    push @ref_spkr_mds, $md;
	}
    }

    foreach $spkr (keys %$sys_mds) {
	foreach $md (@{$sys_mds->{$spkr}}) {
	    push @sys_spkr_mds, $md;
	}
    }

    map_metadata_to_words (\@sys_spkr_mds, $sys_wds, \@ref_spkr_mds, $ref_wds);
}

#################################

sub map_metadata_to_words {

    my ($sys_mds, $sys_wds, $ref_mds, $ref_wds) = @_;

#map system output metadata times to ref words
    foreach my $md (@$sys_mds) {
	if ($opt_w) { #adjust times/words to agree with ref-sys word alignment
	    $md->{RTBEG} = adjust_sys_time_to_ref ($md->{TBEG}, $sys_wds);
	    $md->{RTEND} = adjust_sys_time_to_ref ($md->{TEND}, $sys_wds);
	}
	else { #map system output metadata event to reference data normally
	    $md->{RTBEG} = $md->{TBEG};
	    $md->{RTEND} = $md->{TEND};
	}
	$md->{RTDUR} = $md->{RTEND} - $md->{RTBEG};
	$md->{RTMID} = $md->{RTBEG} + $md->{RTDUR}/2;
	($md->{RWBEG}, $md->{RWEND}) = md_ref_word_indices ($md, $ref_wds);
	$md->{RWDUR} = $md->{RWEND} - $md->{RWBEG} + 1;
    }

#map reference metadata times to ref words
    foreach my $md (@$ref_mds) {
	($md->{WBEG}, $md->{WEND}) = md_word_indices ($md, $ref_wds);
	$md->{WDUR} = $md->{WEND} - $md->{WBEG} + 1;
	next if ($md->{WDUR} > 0 or
		 $md->{TYPE} =~ /^(IP|CB)$/);
        next if (not $opt_W and not $opt_w and $md->{TYPE} eq "SPEAKER");
	warn "\nWARNING:  reference metadata event subsumes no reference words\n"
	    ."    file='$md->{FILE}', chnl='$md->{CHNL}', tbeg='$md->{TBEG}',"
		." tend='$md->{TEND}', type='$md->{TYPE}', subtype='$md->{SUBT}'\n";
    }

#friendly (unused) check of system metadata times versus sys words
    return unless $opt_w;
    foreach my $md (@$sys_mds) {
	(my $wbeg, my $wend) = md_word_indices ($md, $sys_wds);
	next if ($wend - $wbeg >= 0 or
		 $md->{TYPE} =~ /^(IP|CB)$/);
	warn "\nWARNING:  system output metadata event subsumes no system output words\n"
	    ."    file='$md->{FILE}', chnl='$md->{CHNL}', tbeg='$md->{TBEG}',"
		." tend='$md->{TEND}', type='$md->{TYPE}', subtype='$md->{SUBT}'\n";
    }
}

#################################

sub adjust_sys_time_to_ref {

    my ($ts, $sys_wds) = @_;
    my ($ts1, $ts2, $tr, $tr1, $tr2, $ws1, $ws2, $ref_wd);

#given a time in the system output, find the time in the reference
#that harmonizes with the alignment of system output words

#find the nearest right reference anchor point
    $ws2 = 0;
    $ws2++ while ($ws2 < @$sys_wds and
		  ($sys_wds->[$ws2]{TEND} < $ts or
		   not defined $sys_wds->[$ws2]{MAPPTR}));
    if ($ws2 < @$sys_wds) {
	$ref_wd = $sys_wds->[$ws2]{MAPPTR};
	($ts2, $tr2) = $sys_wds->[$ws2]{TBEG} < $ts ?
	    ($sys_wds->[$ws2]{TEND}, $ref_wd->{TEND}) :
	    ($sys_wds->[$ws2]{TBEG}, $ref_wd->{TBEG});
    }

#find the nearest left reference anchor point
    $ws1 = min($ws2, @$sys_wds-1);
    $ws1-- while ($ws1 >= 0 and 
		  ($sys_wds->[$ws1]{TBEG} > $ts or
		   not defined $sys_wds->[$ws1]{MAPPTR}));
    if ($ws1 >= 0) {
	$ref_wd = $sys_wds->[$ws1]{MAPPTR};
	($ts1, $tr1) = $sys_wds->[$ws1]{TEND} > $ts ? 
	    ($sys_wds->[$ws1]{TBEG}, $ref_wd->{TBEG}) :
	    ($sys_wds->[$ws1]{TEND}, $ref_wd->{TEND});
    }

#make adjustment
    $tr = (($ws1 < 0 and $ws2 >= @$sys_wds) ? $ts               : #no adjustment possible
	   ($ws1 < 0)                       ? $tr2 + ($ts-$ts2) : #extrapolate left without scale change
	   ($ws2 >= @$sys_wds)              ? $tr1 + ($ts-$ts1) : #extrapolate right without scale change
	   ($ts == $ts1)                    ? $tr1              : #no interpolation necessary
	   $tr1 + ($ts-$ts1)*($tr2-$tr1)/($ts2-$ts1));            #normal interpolation
    return $tr;
}

#################################

sub md_word_indices {

    my ($md, $wds) = @_;

#find the word indices of the first and last words with midpoints inside the metadata event
    my $i = 0;
    $i++ while ($i<@$wds and ($wds->[$i]{TMID}) < $md->{TBEG});
    my $j = max($i-1,0);
    $j++ while ($j<@$wds and ($wds->[$j]{TMID}) <= $md->{TEND});
    return ($i, --$j);
}

#################################

sub md_ref_word_indices {

    my ($md, $wds) = @_;

#find the word indices of the first and last words with midpoints inside the metadata event
    my $i = 0;
    $i++ while ($i<@$wds and ($wds->[$i]{TMID}) < $md->{RTBEG});
    my $j = max($i-1,0);
    $j++ while ($j<@$wds and ($wds->[$j]{TMID}) <= $md->{RTEND});
    return ($i, --$j);
}

#################################

sub align_data {

    my ($refs, $syss, $spkr, $scorer, $max_delta_score) = @_;
    my ($ref, $sys, $prev_ref, $path, $ref_path);
    my ($ref_index, $sys_index, $index, $pruning_threshold);
    my ($score, $path_score, $best_score, %cum_insertion_score);

#compute cumulative insertion score for sys output
    $cum_insertion_score{-1} = 0;
    for ($sys_index=0; $sys_index<@$syss; $sys_index++) {
	$sys = $syss->[$sys_index];
	$cum_insertion_score{$sys_index} = $cum_insertion_score{$sys_index-1};
	$cum_insertion_score{$sys_index} += &$scorer (undef, $sys);
    }

#find the best path by incremental optimization through the ref transcription
    $prev_ref->{PATHS}{-1}{SCORE} = 0;
    for ($ref_index=0; $ref_index<@$refs; $ref_index++) {
	$ref = $refs->[$ref_index];
	next if $spkr and $ref->{SPKR} ne $spkr;

#find best score and compute pruning threshold
	$best_score = undef;
	foreach $index (keys %{$prev_ref->{PATHS}}) {
	    $path_score = $prev_ref->{PATHS}{$index}{SCORE} +
		$cum_insertion_score{@$syss-1}-$cum_insertion_score{$index};
	    $best_score = $path_score if not defined $best_score or $best_score < $path_score;
	}
	$pruning_threshold = $best_score - $max_delta_score;
	
#extend paths with scores above pruning threshold
	foreach $index (keys %{$prev_ref->{PATHS}}) {
	    $path_score = $prev_ref->{PATHS}{$index}{SCORE} +
		$cum_insertion_score{@$syss-1}-$cum_insertion_score{$index};
	    next unless $path_score > $pruning_threshold;
	    $ref->{PATHS}{$index}{PATHPTR} = $index;
	    $ref->{PATHS}{$index}{PREVREF} = $prev_ref;
	    $ref->{PATHS}{$index}{SCORE} = $prev_ref->{PATHS}{$index}{SCORE} +
		&$scorer ($ref, undef);
	}

#compare the current ref event to all sys events
	for ($sys_index=0; $sys_index<@$syss; $sys_index++) {
	    $sys = $syss->[$sys_index];
	    $score = &$scorer ($ref, $sys);
	    next unless defined $score;

#update each path for this {ref, sys} match
	    foreach $index (sort {$a<=>$b} keys %{$prev_ref->{PATHS}}) {
		next unless $index < $sys_index;
		$path_score = $score + $prev_ref->{PATHS}{$index}{SCORE} +
		    $cum_insertion_score{$sys_index-1}-$cum_insertion_score{$index};
		if (not defined $ref->{PATHS}{$sys_index}
		    or $path_score > $ref->{PATHS}{$sys_index}{SCORE}) {
		    $ref->{PATHS}{$sys_index}{SCORE} = $path_score;
		    $ref->{PATHS}{$sys_index}{PREVREF} = $prev_ref;
		    $ref->{PATHS}{$sys_index}{PATHPTR} = $index;
		    $ref->{PATHS}{$sys_index}{SYSPTR} = $sys;
		}
	    }
	}
	$prev_ref=$ref;
    }

#add insertion score for remaining unmapped sys events
    foreach $index (sort {$a<=>$b} keys %{$prev_ref->{PATHS}}) {
	$prev_ref->{PATHS}{$index}{SCORE} +=
	    $cum_insertion_score{@$syss-1}-$cum_insertion_score{$index} if $index < @$syss-1;
    }
}

#################################

sub md_score {

    my ($ref_md, $sys_md) = @_;
    my ($beg, $end, $overlap, $ref_beg, $sys_beg, $md_dur);
    my $subtype_bonus = 1.1; #multiplicative bonus for matching subtypes
    my $endword_bonus = 1.001; #multiplicative bonus for matching boundaries

    return 0 unless defined $ref_md and defined $sys_md;

    if ($opt_W) { #compute md mapping score as ref-sys overlap in (ref) words
	$ref_beg = $ref_md->{WBEG};
	$sys_beg = $sys_md->{RWBEG};
	if ($ref_md->{TYPE} eq "SU") {
	    $ref_beg = max($ref_beg, $ref_md->{WEND}-($su_extent_limit-1));
	    $sys_beg = max($sys_beg, $sys_md->{RWEND}-($su_extent_limit-1));
	}
	$beg = max($ref_beg, $sys_beg);
	$end = min($ref_md->{WEND}, $sys_md->{RWEND});
	$overlap = $end - $beg + 1;
	$md_dur = $ref_md->{WEND} - $ref_beg + 1;
    }
    else { #compute md mapping score as ref-sys overlap in time
	$ref_beg = $ref_md->{TBEG};
	$sys_beg = $sys_md->{RTBEG};
	if ($ref_md->{TYPE} eq "SU") {
	    $ref_beg = max($ref_beg, $ref_md->{TEND}-$su_extent_limit);
	    $sys_beg = max($sys_beg, $sys_md->{RTEND}-$su_extent_limit);
	}
	$beg = max($ref_beg, $sys_beg);
	$end = min($ref_md->{TEND}, $sys_md->{RTEND});
	$overlap = $end - $beg;
	$md_dur = $ref_md->{TEND} - $ref_beg;
    }
    $overlap += $epsilon if $ref_md->{TYPE} =~ /^(IP|CB)$/;
    $overlap += $md_gap;
    return undef if $overlap < 0;
    $overlap *= $subtype_bonus if $ref_md->{SUBT} eq $sys_md->{SUBT};
    $overlap *= $endword_bonus if $ref_md->{WEND} eq $sys_md->{RWEND};
    return $overlap if $md_dur+$md_gap < max($md_dur,$epsilon);
    return $overlap * max($md_dur,$epsilon)/($md_dur+$md_gap);
}

#################################

sub trace_best_path {

    my ($refs, $syss, $spkr) = @_;
    my ($ref, $path, $pathptr, $best_score, $prev_ref, $ref_index, $index, $sys);

#find the last word for the selected channel and speaker
    return unless @$refs and @$syss;
    $ref_index = @$refs-1;
    $ref_index-- while (defined $spkr and $refs->[$ref_index]{SPKR} ne $spkr);
    $spkr = "ALL" unless defined $spkr;

#identify the best path for the selected ending word
    $ref = $refs->[$ref_index];
    undef $best_score;
    foreach $index (sort {$a<=>$b} keys %{$ref->{PATHS}}) {
	$path = $ref->{PATHS}{$index};
	if (not defined $best_score or $path->{SCORE} > $best_score) {
	    $best_score = $path->{SCORE};
	    $pathptr = $path->{PATHPTR};
	    $prev_ref = $path->{PREVREF};
	    $sys = $path->{SYSPTR};
	}
    }
    if (defined $sys) {
	$sys->{SPKRS}{$spkr}{REFPTR} = $ref;
	$sys->{MAPPTR} = $ref;
	$ref->{MAPPTR} = $sys;
    }

#trace the path back 
    while ($pathptr != -1) {
	$ref = $prev_ref;
	$path = $ref->{PATHS}{$pathptr};
	$pathptr = $path->{PATHPTR};
	$prev_ref = $path->{PREVREF};
	next unless defined $path->{SYSPTR};
	$sys = $path->{SYSPTR};
	$sys->{SPKRS}{$spkr}{REFPTR} = $ref;
	$sys->{MAPPTR} = $ref;
	$ref->{MAPPTR} = $sys;
    }
}

#################################

sub delta_metadata_error_words {

#accumulates the number of metadata error words difference
#between ref beg/end point of metadata event and sys beg/end point of metadata event

    my ($location, $ref_index, $sys_index, $ref_wds) = @_;

    my $dw = 0;
    my $index = min($ref_index,$sys_index);
    my $istop = max($ref_index,$sys_index);
    while ($index != $istop) {
	$index++ if $location eq "END";
	$dw++ if $index >= 0 and $index < @$ref_wds and $ref_wds->[$index]{SCOREABLE};
	$index++ if $location eq "BEG";
    }
    return $sys_index > $ref_index ? $dw : 0-$dw;
}

#################################

sub print_path_score {

    my ($ref, $sys, $ref_count, $err_count, $err_type) = @_;

#print header
    unless (defined $ref or defined $sys) {
	printf " ref del ins sub %16.16s %-7s%8s%8s %-12.12s", "REF:  token", "type",
	    "tbeg", "tend", "speaker";
	printf " %16.16s %-7s  %7s%8s %8s%8s %-12.12s\n", "SYS:  token", "type",
	    "Rtbeg", "Rtend", "tbeg", "tend", "sys-speaker" if $opt_w;
	printf " %16.16s %-7s%8s%8s %-12.12s\n", "SYS:  token", "type",
	    "tbeg", "tend", "speaker" unless $opt_w;
	return;
    }

#print ref
    my %errors = (REF=>"-", DEL=>"-", INS=>"-", SUB=>"-");
    $errors{REF} = $ref_count if defined $ref_count;
    $errors{$err_type} = $err_count if defined $err_type;
    printf "%4s%4s%4s%4s", $errors{REF}, $errors{DEL}, $errors{INS}, $errors{SUB};

    if (defined $ref) {
	printf " %16.16s %-7s%8.2f%8.2f %-12.12s", $ref->{TYPE} =~ /^(LEXEME|NON-LEX|NON-SPEECH)$/ ?
	    ($ref->{WORD}, $ref->{WTYP}) : ($ref->{SUBT}, $ref->{TYPE}), $ref->{TBEG}, $ref->{TEND}, $ref->{SPKR};
    }
    else {
	printf " %16.16s %-7s%8s%8s %-12.12s", "---", "---", "--- ", "--- ", "---";
    }

#print sys
    if ($opt_w) {
	if (defined $sys) {
	    printf " %16.16s %-7s (%7.2f%8.2f)%8.2f%8.2f %-12.12s\n", $sys->{TYPE} =~ /^(LEXEME|NON-LEX|NON-SPEECH)$/ ?
		($sys->{WORD}, $sys->{WTYP}) : ($sys->{SUBT}, $sys->{TYPE}), $sys->{RTBEG}, $sys->{RTEND}, $sys->{TBEG}, $sys->{TEND}, $sys->{SPKR};
	}
	else {
	    printf " %16.16s %-7s (%7s%8s)%8s%8s %-12.12s\n", "---", "---", "--- ", "--- ", "--- ", "--- ", "---";
	}
    }
    else {
	if (defined $sys) {
	    printf " %16.16s %-7s%8.2f%8.2f %-12.12s\n", $sys->{TYPE} =~ /^(LEXEME|NON-LEX|NON-SPEECH)$/ ?
		($sys->{WORD}, $sys->{WTYP}) : ($sys->{SUBT}, $sys->{TYPE}), $sys->{TBEG}, $sys->{TEND}, $sys->{SPKR};
	}
	else {
	    printf " %16.16s %-7s%8s%8s %-12.12s\n", "---", "---", "--- ", "--- ", "---";
	}
    }	
}

#################################

sub score_metadata_path {

    my ($type, $file, $chnl, $ref_mds, $sys_mds, $ref_wds) = @_;
    my ($ref_md, @sys_mds, $sys_index, $sys_md, $md, $spkr, $iw);
    my (%count, $ref_count, $err_count, $ref_wd, $dw);

    print "\n$type alignment and scoring details for channel $chnl of file $file\n" if $opt_D;
    print_path_score () if $opt_D;

#tabulate boundary/depod errors
    tag_ref_words_with_metadata_info ($ref_mds, $ref_wds, "REF");
    tag_ref_words_with_metadata_info ($sys_mds, $ref_wds, "SYS");
    for ($iw=0; $iw<@$ref_wds; $iw++) {
	$ref_wd = $ref_wds->[$iw];
	next unless $ref_wd->{SCOREABLE} or $type =~ /^(IP|SU)$/;
	my $nref = my $nsys = my $nins = my $ncor = 0;
	foreach my $subtype (keys %{$md_subtypes{$type}}) {
	    my $nr = $ref_wd->{"REF-$type"}{$subtype}{MAP};
	    my $nm = $ref_wd->{"REF-$type"}{$subtype}{NOT};
	    my $ns = $ref_wd->{"SYS-$type"}{$subtype}{MAP};
	    my $ni = $ref_wd->{"SYS-$type"}{$subtype}{NOT};
	    $nref += $nr if $nr;
	    $nref += $nm if $nm;
	    $nsys += $ns if $ns;
	    $nins += $ni if $ni;
	    $ncor += min($nr,$ns) if $nr and $ns;
	}
	$count{WORDS}{REF} += $nref;
	$count{WORDS}{DEL} += max($nref-$nsys,0);
	$count{WORDS}{INS} += max($nsys-$nref,0) + ($nins ? $nins : 0);
	$count{WORDS}{SUB} += min($nref,$nsys) - $ncor;
    }

#tabulate beg/end word offset errors
    foreach $ref_md (@$ref_mds) {
	next unless ($sys_md = $ref_md->{MAPPTR});
	$dw = delta_metadata_error_words ("BEG", $ref_md->{WBEG}, $sys_md->{RWBEG}, $ref_wds);
	$count{WORD_OFFSET}{WBEG}{$dw}++;
	$dw = delta_metadata_error_words ("END", $ref_md->{WEND}, $sys_md->{RWEND}, $ref_wds);
	$count{WORD_OFFSET}{WEND}{$dw}++;
    }

#tabulate detection errors
    @sys_mds = @$sys_mds;
    $sys_md = shift @sys_mds;
    foreach $ref_md (@$ref_mds) {
	$spkr = $ref_md->{SPKR};
	$ref_count = md_err_count ($ref_md, undef);
	$count{REF}{$spkr}{$ref_md->{SUBT}} += $ref_count if defined $ref_count;
	if ($ref_md->{MAPPTR}) {
	    while ($sys_md and
		   $sys_md ne $ref_md->{MAPPTR}) {
		printf "%sUNEXPECTED MAPPED SYS MD:  %16s %-7s%8.2f%8.2f %-16s\n",
		    " "x44, $sys_md->{SUBT}, $sys_md->{TYPE}, $sys_md->{TBEG},
		    $sys_md->{TEND}, $sys_md->{SPKR} if $sys_md->{MAPPTR};
		$err_count = md_err_count (undef, $sys_md);
		$count{INS}{ref_spkr_of_md($sys_md,$ref_wds)}{$sys_md->{SUBT}} += $err_count;
		print_path_score (undef, $sys_md, 0, $err_count, "INS") if $opt_D;
		$sys_md = shift @sys_mds;
	    }
	    if ($sys_md) {
		$err_count = md_err_count ($ref_md, $sys_md);
		$count{SUB}{$spkr}{$ref_md->{SUBT}} += $err_count;
		$count{CONFUSION}{$spkr}{$ref_md->{SUBT}}{$sys_md->{SUBT}} += $ref_count;
		print_path_score ($ref_md, $sys_md, $ref_count, $err_count, "SUB") if $opt_D;
		$sys_md = shift @sys_mds;
	    }
	    else {
		printf "%sSYS MD NOT FOUND FOR REF MD:   %16s %-7s%8.2f%8.2f %-16s\n",
		    " "x40, $ref_md->{SUBT}, $ref_md->{TYPE}, $ref_md->{TBEG},
		    $ref_md->{TEND}, $ref_md->{SPKR} if $ref_md->{MAPPTR};
	    }
	}
	else {
	    $err_count = md_err_count ($ref_md, undef);
	    $count{DEL}{$spkr}{$ref_md->{SUBT}} += $err_count;
	    print_path_score ($ref_md, undef, $ref_count, $err_count, "DEL") if $opt_D;
	}
    }
    while ($sys_md) {
	printf "%sUNEXPECTED MAPPED SYS MD:  %16s %-7s%8.2f%8.2f %-16s\n",
	    " "x44, $sys_md->{SUBT}, $sys_md->{TYPE}, $sys_md->{TBEG},
	    $sys_md->{TEND}, $sys_md->{SPKR} if $sys_md->{MAPPTR};
	$err_count = md_err_count (undef, $sys_md);
	$count{INS}{ref_spkr_of_md($sys_md,$ref_wds)}{$sys_md->{SUBT}} += $err_count;
	print_path_score (undef, $sys_md, 0, $err_count, "INS") if $opt_D;
	$sys_md = shift @sys_mds;
    }
    return {%count};
}

#################################

sub md_err_count {

    my ($ref_md, $sys_md) = @_;

    return 1 if (not defined $sys_md         or not defined $ref_md         or
		 not defined $sys_md->{TYPE} or not defined $ref_md->{TYPE} or
		 not defined $sys_md->{SUBT} or not defined $ref_md->{SUBT} or
		 $sys_md->{TYPE} ne $ref_md->{TYPE} or
		 $sys_md->{SUBT} ne $ref_md->{SUBT});
    return 0;
}

#################################

sub ref_spkr_of_md {

    my ($md, $ref_wds) = @_;
    my $spkr;

    for (my $index =min($md->{RWBEG},$md->{RWEND});
	    $index<=max($md->{RWBEG},$md->{RWEND}); $index++) {
	next unless $index >= 0 and $index < @$ref_wds;
	$spkr = $ref_wds->[$index]{SPKR} unless $spkr;
	return "unknown" unless $ref_wds->[$index]{SPKR} eq $spkr;
    }
    return defined $spkr ? $spkr : "unknown";
}

#################################

sub score_word_path {

    my ($file, $chnl, $ref_wds, $sys_wds) = @_;
    my ($ref_wrd, @sys_wds, $sys_wrd);
    my ($ref_count, $err_count);

    print "\nWord alignment and scoring details for channel $chnl of file $file\n";
    print_path_score ();

#tabulate errors
    @sys_wds = @$sys_wds;
    $sys_wrd = shift @sys_wds;
    foreach $ref_wrd (@$ref_wds) {
	$ref_count = ref_count($ref_wrd);
	if ($ref_wrd->{MAPPTR}) {
	    while ($sys_wrd and
		   $sys_wrd ne $ref_wrd->{MAPPTR}) {
		printf "%71s%16s %-7s%s%8.2f%8.2f %-16s\n", "UNEXPECTED MAPPED SYS WORD:",
		    $sys_wrd->{WORD}, $sys_wrd->{WTYP}, " "x18, $sys_wrd->{TBEG},
		    $sys_wrd->{TDUR}, $sys_wrd->{SPKR} if $sys_wrd->{MAPPTR};
		$err_count = wd_err_count(undef, $sys_wrd);
		print_path_score (undef, $sys_wrd, 0, $err_count, "INS");
		$sys_wrd = shift @sys_wds;
	    }
	    if ($sys_wrd) {
		$err_count = wd_err_count($ref_wrd, $sys_wrd);
		print_path_score ($ref_wrd, $sys_wrd, $ref_count, $err_count, "SUB");
		$sys_wrd = shift @sys_wds;
	    }
	    else {
		printf "%71s%16s %-7s%s%8.2f%8.2f %-16s\n", "SYS WRD NOT FOUND FOR REF WRD:",
		    $ref_wrd->{WORD}, $ref_wrd->{WTYP}, " "x18, $ref_wrd->{TBEG},
		    $ref_wrd->{TDUR}, $ref_wrd->{SPKR} if $ref_wrd->{MAPPTR};
	    }
	}
	else {
	    $err_count = wd_err_count($ref_wrd, undef);
	    print_path_score ($ref_wrd, undef, $ref_count, $err_count, "DEL");
	}
    }
    while ($sys_wrd) {
	printf "%71s%16s %-7s%8.2f%8.2f %-16s\n", "UNEXPECTED MAPPED SYS WORD:",
	    $sys_wrd->{WORD}, $sys_wrd->{WTYP}, $sys_wrd->{TBEG},
	    $sys_wrd->{TDUR}, $sys_wrd->{SPKR} if $sys_wrd->{MAPPTR};
	$err_count = wd_err_count(undef, $sys_wrd);
	print_path_score (undef, $sys_wrd, 0, $err_count, "INS");
	$sys_wrd = shift @sys_wds;
    }
}

#################################

sub date_time_stamp {

    my ($sec, $min, $hour, $mday, $mon, $year, $wday, $yday, $isdst) = localtime();
    my @months = qw(Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec);
    my ($date, $time);

    $time = sprintf "%2.2d:%2.2d:%2.2d", $hour, $min, $sec;
    $date = sprintf "%4.4s %3.3s %s", 1900+$year, $months[$mon], $mday;
    return ($date, $time);
}

#################################

sub max {

    my ($max, $next);

    return unless defined ($max=pop);
    while (defined ($next=pop)) {
	$max = $next if $next > $max;
    }
    return $max;
}

#################################

sub min {

    my ($min, $next);

    return unless defined ($min=pop);
    while (defined ($next=pop)) {
	$min = $next if $next < $min;
    }
    return $min;
}

#################################

sub score_speaker_diarization {

    my ($file, $chnl, $ref_spkr_data, $sys_spkr_data, $ref_wds, $uem_eval, $rttm_data) = @_;
    my ($uem_score, $ref_eval, $sys_eval, $spkr_overlap, $spkr_map);
    my ($eval_segs, $score_segs, %stats, @ref_wds, $wrd, $ref_spkr, $sys_spkr);
    my ($nref, $nsys, $nmap, $spkr, $seg, $type, $spkr_info, $noscore_nl);

    $stats{EVAL_WORDS} = $stats{SCORED_WORDS} =	$stats{MISSED_WORDS} = $stats{ERROR_WORDS} = $epsilon;
    @ref_wds = @$ref_wds;
    $wrd = shift @ref_wds;
    foreach $seg (@$uem_eval) {
	$stats{EVAL_TIME} += $seg->{TEND}-$seg->{TBEG};
	$wrd = shift @ref_wds while ($wrd and $wrd->{TMID} < $seg->{TBEG});
	while ($wrd and $wrd->{TMID} <= $seg->{TEND}) {
	    $stats{EVAL_WORDS}++;
	    $wrd = shift @ref_wds;
	}
    }

    $eval_segs = create_speaker_segs ($uem_eval, $ref_spkr_data, $sys_spkr_data);
    foreach $seg (@$eval_segs) {
	foreach $ref_spkr (keys %{$seg->{REF}}) {
	    $spkr_info->{REF}{$ref_spkr}{TIME} += $seg->{TDUR};
	    $spkr_info->{REF}{$ref_spkr}{TYPE} = $ref_spkr_data->{$ref_spkr}[0]{SUBT};
	}
	foreach $sys_spkr (keys %{$seg->{SYS}}) {
	    $spkr_info->{SYS}{$sys_spkr}{TIME} += $seg->{TDUR};
	    $spkr_info->{SYS}{$sys_spkr}{TYPE} = $sys_spkr_data->{$sys_spkr}[0]{SUBT};
	}
	next unless keys %{$seg->{REF}} > 0;
	$stats{EVAL_SPEECH} += $seg->{TDUR};
	foreach $ref_spkr (keys %{$seg->{REF}}) {
	    foreach $sys_spkr (keys %{$seg->{SYS}}) {
		$spkr_overlap->{$ref_spkr}{$sys_spkr} += $seg->{TDUR};
	    }
	}
    }
    $speaker_map{$file}{$chnl} = $spkr_map = map_speakers ($spkr_overlap)
	if defined $spkr_overlap;
    print_speaker_map ($spkr_map, $spkr_overlap) if $opt_m;
    update_speaker_map_file ($spkr_map, $spkr_overlap, $file, $chnl, $opt_M) if $opt_M;

    $uem_score = $collar > 0 ? add_collars_to_uem ($uem_eval, $ref_spkr_data) : $uem_eval;
    $uem_score = add_exclusion_zones_to_uem ($noscore_sd, $uem_score, $rttm_data);
    $noscore_nl->{"NON-LEX"} = $noscore_sd->{"NON-LEX"};
    $uem_score = add_exclusion_zones_to_uem ($noscore_nl, $uem_score, $rttm_data, $max_extend);
    $uem_score = exclude_overlapping_speech_from_uem ($uem_score, $rttm_data) if $opt_1;
    tag_scoreable_words ($ref_wds, $uem_score);
    $score_segs = create_speaker_segs ($uem_score, $ref_spkr_data, $sys_spkr_data);
    print_speaker_segs ($score_segs, $file, $chnl) if $opt_v;
    ($stats{TYPE}{NSPK}) = speaker_mapping_scores ($spkr_map, $spkr_info);
    score_speaker_segments (\%stats, $score_segs, $ref_wds, $spkr_map, $spkr_info);
    return {%stats};
}

#################################

sub speaker_mapping_scores {

    my ($spkr_map, $spkr_info) = @_;
    my ($ref_spkr, $ref_type, $sys_spkr, $sys_type, %imap, %stats);

    foreach $ref_spkr (keys %{$spkr_info->{REF}}) {
	next unless $spkr_info->{REF}{$ref_spkr}{TIME};
	$ref_type = $spkr_info->{REF}{$ref_spkr}{TYPE};
	$stats{REF}{$ref_type}++;
	$sys_spkr = $spkr_map->{$ref_spkr};
	$sys_type = defined $sys_spkr ? $spkr_info->{SYS}{$sys_spkr}{TYPE} : $miss_name;
	$stats{JOINT}{$ref_type}{$sys_type}++;
	$imap{$sys_spkr} = $ref_spkr if defined $sys_spkr;
    }
    foreach $sys_spkr (keys %{$spkr_info->{SYS}}) {
	next unless $spkr_info->{SYS}{$sys_spkr}{TIME};
	$sys_type = $spkr_info->{SYS}{$sys_spkr}{TYPE};
	$stats{SYS}{$sys_type}++;
	$stats{JOINT}{$fa_name}{$sys_type}++
	    unless defined $imap{$sys_spkr};
    }
    return {%stats};
}

#################################

sub score_speaker_segments {

    my ($stats, $score_segs, $ref_wds, $spkr_map, $spkr_info) = @_;
    my ($ref_spkr, $ref_type, $sys_spkr, $sys_type, %type_stats);
    my (@ref_wds, $wrd, $seg, $seg_dur, $nref, $nsys);

    @ref_wds = @$ref_wds;
    $wrd = shift @ref_wds;
    foreach $seg (@$score_segs) {
	$seg_dur = $seg->{TDUR};
	$stats->{SCORED_TIME} += $seg_dur;
	$nref = keys %{$seg->{REF}};
	$nsys = keys %{$seg->{SYS}};
	$stats->{SCORED_SPEECH} += $nref ? $seg_dur : 0;
	$stats->{MISSED_SPEECH} += ($nref and not $nsys) ? $seg_dur : 0;
	$stats->{FALARM_SPEECH} += ($nsys and not $nref) ? $seg_dur : 0;
	$stats->{SCORED_SPEAKER} += $seg_dur*$nref;
	$stats->{MISSED_SPEAKER} += $seg_dur*max($nref-$nsys,0);
	$stats->{FALARM_SPEAKER} += $seg_dur*max($nsys-$nref,0);

	my $scored_wrds = my $missed_wrds = my $error_wrds = 0;
	$wrd = shift @ref_wds while ($wrd and $wrd->{TMID} < $seg->{TBEG});
	while ($wrd and $wrd->{TMID} <= $seg->{TEND}) {
	    next unless $wrd->{SCOREABLE};
	    $scored_wrds++;
	    $missed_wrds++ if not $nsys;
	    $error_wrds++ unless speakers_match ($seg->{REF}, $seg->{SYS}, $spkr_map);
	    $wrd = shift @ref_wds;
	}
	$stats->{SCORED_WORDS} += $scored_wrds;
	$stats->{MISSED_WORDS} += $missed_wrds;
	$stats->{ERROR_WORDS} += $error_wrds;

	my $nmap = 0, my %num_types;
	foreach $ref_spkr (keys %{$seg->{REF}}) {
	    $ref_type = $spkr_info->{REF}{$ref_spkr}{TYPE};
	    $num_types{REF}{$ref_type}++;
	    $sys_spkr = $spkr_map->{$ref_spkr};
	    $nmap++ if defined $sys_spkr and defined $seg->{SYS}{$sys_spkr};
	}
	$stats->{SPEAKER_ERROR} += $seg_dur*(min($nref,$nsys) - $nmap);

	foreach $sys_spkr (keys %{$seg->{SYS}}) {
	    $sys_type = $spkr_info->{SYS}{$sys_spkr}{TYPE};
	    $num_types{SYS}{$sys_type}++;
	}
	foreach $ref_type (keys %{$num_types{REF}}) {
	    $nref = $num_types{REF}{$ref_type};
	    $type_stats{REF}{$ref_type} += $nref*$seg_dur;
	    foreach $sys_type (keys %{$num_types{SYS}}) {
		$nsys = $num_types{SYS}{$sys_type};
		$type_stats{JOINT}{$ref_type}{$sys_type} += min($nref,$nsys)*$seg_dur;
	    }
	    $type_stats{JOINT}{$ref_type}{$miss_name} += max($nref-$nsys,0)*$seg_dur;
	}
	foreach $sys_type (keys %{$num_types{SYS}}) {
	    $nsys = $num_types{SYS}{$sys_type};
	    $type_stats{SYS}{$sys_type} += $nsys*$seg_dur;
	    $type_stats{JOINT}{$fa_name}{$sys_type} += max($nsys-$nref,0)*$seg_dur;
	}
    }
    $stats->{TYPE}{TIME} = {%type_stats};
}

#################################

sub speakers_match {

    my ($ref_spkrs, $sys_spkrs, $spkr_map) = @_;

    return 0 unless keys %$ref_spkrs == keys %$sys_spkrs;
    foreach my $ref_spkr (keys %$ref_spkrs) {
	return 0 unless (defined $spkr_map->{$ref_spkr} and
			 defined $sys_spkrs->{$spkr_map->{$ref_spkr}});
    }
    return 1;
}

#################################

sub add_collars_to_uem {

    my ($uem_eval, $ref_data) = @_;
    my (@events, $event, $uem, $uem_score, $spkr, $spkr_seg, $tbeg, $evaluate);

    foreach $uem (@$uem_eval) {
	push @events, {EVENT => "BEG", TIME => $uem->{TBEG}};
	push @events, {EVENT => "END", TIME => $uem->{TEND}};
    }
#add no-score collars
    foreach $spkr (keys %$ref_data) {
	foreach $spkr_seg (@{$ref_data->{$spkr}}) {
	    push @events, {EVENT => "END", TIME => $spkr_seg->{TBEG}-$collar};
	    push @events, {EVENT => "BEG", TIME => $spkr_seg->{TBEG}+$collar};
	    push @events, {EVENT => "END", TIME => $spkr_seg->{TEND}-$collar};
	    push @events, {EVENT => "BEG", TIME => $spkr_seg->{TEND}+$collar};
	}
    }
    @events = sort {($a->{TIME} < $b->{TIME} ? -1 :
		     ($a->{TIME} > $b->{TIME} ? 1 :
		      $a->{EVENT} eq "END"))} @events;
    $evaluate = 0;
    foreach $event (@events) {
	if ($event->{EVENT} eq "BEG") {
	    $evaluate++;
	    $tbeg = $event->{TIME} if $evaluate == 1;
	}
	else {
	    $evaluate--;
	    push @$uem_score, {TBEG => $tbeg, TEND => $event->{TIME}}
	        if $evaluate == 0 and $event->{TIME} > $tbeg;
	}
    }
    return $uem_score;
}

#################################

sub exclude_overlapping_speech_from_uem {

    my ($uem_data, $rttm_data) = @_;
    my ($token, @spkr_events, $event, $spkr_cnt, $tbeg_overlap, $uem, @events, $uem_ex);

#overlapping speech computed from SPEAKER data
    foreach $token (@$rttm_data) {
	next unless ($token->{TYPE} eq "SPEAKER" and
		     $token->{TDUR} > 0);
	push @spkr_events, {EVENT => "BEG", TIME => $token->{TBEG}, SPKR => $token->{SPKR}};
	push @spkr_events, {EVENT => "END", TIME => $token->{TEND}, SPKR => $token->{SPKR}};
    }
    @spkr_events = sort {($a->{TIME} < $b->{TIME} ? -1 :
			  ($a->{TIME} > $b->{TIME} ? 1 :
			   $a->{EVENT} eq "BEG"))} @spkr_events;

#create noscore zones
    foreach $event (@spkr_events) {
	if ($event->{EVENT} eq "BEG") {
	    next unless ++$spkr_cnt == 2;
	    $tbeg_overlap = $event->{TIME};
	}
	else {
	    next unless --$spkr_cnt == 1;
	    push @events, {TYPE => "NSZ", EVENT => "BEG", TIME => $tbeg_overlap};
	    push @events, {TYPE => "NSZ", EVENT => "END", TIME => $event->{TIME}};
	}
    }
	
#merge noscore zones with UEM data
    foreach $uem (@$uem_data) {
	next unless $uem->{TEND}-$uem->{TBEG} > 0;
	push @events, {TYPE => "UEM", EVENT => "BEG", TIME => $uem->{TBEG}};
	push @events, {TYPE => "UEM", EVENT => "END", TIME => $uem->{TEND}};
    }
    @events = sort {($a->{TIME} < $b->{TIME} ? -1 :
		     ($a->{TIME} > $b->{TIME} ? 1 :
		      $a->{EVENT} eq "BEG"))} @events;

    my $tbeg = my $evl_cnt = my $nsz_cnt = my $evaluating = 0;
    foreach $event (@events) {
	$evl_cnt += $event->{EVENT} eq "BEG" ? 1 : -1 if $event->{TYPE} eq "UEM";
	$nsz_cnt += $event->{EVENT} eq "BEG" ? 1 : -1 if $event->{TYPE} eq "NSZ";
	if ($evaluating and
	    ($evl_cnt == 0 or $nsz_cnt > 0) and
	    $event->{TIME} > $tbeg) {
	    push @$uem_ex, {TBEG => $tbeg, TEND => $event->{TIME}};
	    $evaluating = 0;
	}
	elsif ($evl_cnt > 0 and $nsz_cnt == 0) {
	    $tbeg = $event->{TIME};
	    $evaluating = 1;
	}
    }
	    
    return $uem_ex;
}

#################################

sub add_exclusion_zones_to_uem {

    my ($excluded_tokens, $uem_score, $rttm_data, $max_extend) = @_;
    my (@events, $event, $uem, $uem_ex, $spkr, $spkr_seg, $tbeg, $evaluating, $token);
    my (@ns_events, $evl_cnt, $lex_cnt, $nsz_cnt, $tstart, $tstop);
    my ($tbeg_lex, $tbeg_nsz, $tend_lex, $tend_nsz, $tseg);

    return $uem_score unless defined $excluded_tokens and (keys %$excluded_tokens) > 0;

#gather data needed to create noscore zones
    foreach $token (@$rttm_data) {
	if ($token->{TYPE} eq "LEXEME" and
	    not defined $excluded_tokens->{LEXEME}{$token->{SUBT}} and
	    $token->{TDUR} > 0) {
	    push @ns_events, {TYPE => "LEX", EVENT => "BEG", TIME => $token->{TBEG}};
	    push @ns_events, {TYPE => "LEX", EVENT => "END", TIME => $token->{TEND}};
	}
	elsif ($token->{TYPE} eq "SPEAKER" and
	       $token->{TDUR} > 0) {
	    push @ns_events, {TYPE => "SEG", EVENT => "BEG", TIME => $token->{TBEG}};
	    push @ns_events, {TYPE => "SEG", EVENT => "END", TIME => $token->{TEND}};
	}
	elsif (defined $excluded_tokens->{$token->{TYPE}}{$token->{SUBT}} and
	       $token->{TDUR} > 0) {
	    push @ns_events, {TYPE => "NSZ", EVENT => "BEG", TIME => $token->{TBEG}};
	    push @ns_events, {TYPE => "NSZ", EVENT => "END", TIME => $token->{TEND}};
	}
    }
    @ns_events = sort {($a->{TIME} < $b->{TIME} ? -1 :
			($a->{TIME} > $b->{TIME} ? 1 :
			 $a->{EVENT} eq "BEG"))} @ns_events;

#create noscore zones
    $evaluating = 1;
    $max_extend = $epsilon if not $max_extend or $max_extend < $epsilon;
    $tseg = $tbeg_nsz = $tbeg_lex = $tend_nsz = $tend_lex = 0;
    $lex_cnt = $nsz_cnt = 0;
    foreach $event (@ns_events) {
	if ($event->{TYPE} eq "LEX") {
	    if ($event->{EVENT} eq "BEG") {
		$tbeg_lex = $event->{TIME} if $lex_cnt++ == 0;
	    }
	    else {
		$tend_lex = $event->{TIME} if $lex_cnt-- == 1;
	    }
	}
	elsif ($event->{TYPE} eq "NSZ") {
	    if ($event->{EVENT} eq "BEG") {
		$tbeg_nsz = $event->{TIME} if $nsz_cnt++ == 0;
	    }
	    else {
		$tend_nsz = $event->{TIME} if $nsz_cnt-- == 1;
	    }
	}
	elsif ($event->{TYPE} eq "SEG") {
	    $tseg = $event->{TIME};
	}

	if ($evaluating) {
	    next if ($nsz_cnt == 0 or
		     $event->{TYPE} ne "NSZ");
	    $tstop = ($lex_cnt > 0 ? $event->{TIME} :
		      max($tend_lex, $tseg, $event->{TIME}-$max_extend));
	    push @events, {TYPE => "NSZ", EVENT => "BEG", TIME => $tstop};
	    $evaluating = 0;
	}
	elsif ($nsz_cnt == 0 and
	       ($lex_cnt > 0 or
		$event->{TYPE} eq "SEG")) {
	    $tstart = min($tend_nsz+$max_extend, $event->{TIME});
	    push @events, {TYPE => "NSZ", EVENT => "END", TIME => $tstart};
	    $evaluating = 1;
	}
	elsif ($nsz_cnt == 1 and
	       $event->{TYPE} eq "NSZ" and
	       $event->{EVENT} eq "BEG" and
	       $event->{TIME} > $tend_nsz+2*$max_extend) {
	    push @events, {TYPE => "NSZ", EVENT => "END", TIME => $tend_nsz+$max_extend};
	    push @events, {TYPE => "NSZ", EVENT => "BEG", TIME => $event->{TIME}-$max_extend};
	    $evaluating = 0;
	}
    }

#merge noscore zones with UEM data
    foreach $uem (@$uem_score) {
	next unless $uem->{TEND}-$uem->{TBEG} > 0;
	push @events, {TYPE => "UEM", EVENT => "BEG", TIME => $uem->{TBEG}};
	push @events, {TYPE => "UEM", EVENT => "END", TIME => $uem->{TEND}};
    }
    @events = sort {($a->{TIME} < $b->{TIME} ? -1 :
		     ($a->{TIME} > $b->{TIME} ? 1 :
		      $a->{EVENT} eq "BEG"))} @events;
    $evl_cnt = $evaluating = 0;
    foreach $event (@events) {
	$evl_cnt += $event->{EVENT} eq "BEG" ? 1 : -1 if $event->{TYPE} eq "UEM";
	$nsz_cnt += $event->{EVENT} eq "BEG" ? 1 : -1 if $event->{TYPE} eq "NSZ";
	if ($evaluating and
	    ($evl_cnt == 0 or $nsz_cnt > 0) and
	    $event->{TIME} > $tbeg) {
	    push @$uem_ex, {TBEG => $tbeg, TEND => $event->{TIME}};
	    $evaluating = 0;
	}
	elsif ($evl_cnt > 0 and $nsz_cnt == 0) {
	    $tbeg = $event->{TIME};
	    $evaluating = 1;
	}
    }
	    
    return $uem_ex;
}

#################################

sub uem_from_rttm {

    my ($rttm_data) = @_;
    my ($token, $tbeg, $tend);

    ($tbeg, $tend) = (1E30, 0);
    foreach $token (@$rttm_data) {
	($tbeg, $tend) = (min($tbeg,$token->{TBEG}), max($tend,$token->{TEND})) if
	    $token->{TYPE} =~ /^(SEGMENT|SPEAKER|SU|EDIT|FILLER|IP|CB|A\/P|LEXEME|NON-LEX)$/;
    }

    return [{TBEG => $tbeg, TEND => $tend}];
}

#################################

sub create_speaker_segs {

    my ($uem_score, $ref_data, $sys_data) = @_;
    my ($spkr, $seg, @events, $event, $uem, $segments, $tbeg, $tend);
    my ($evaluate, %ref_spkrs, %sys_spkrs, $spkrs);

    foreach $uem (@$uem_score) {
	next unless $uem->{TEND} > $uem->{TBEG}+$epsilon;
	push @events, {TYPE => "UEM", EVENT => "BEG", TIME => $uem->{TBEG}};
	push @events, {TYPE => "UEM", EVENT => "END", TIME => $uem->{TEND}};
    }
    foreach $spkr (keys %$ref_data) {
	foreach $seg (@{$ref_data->{$spkr}}) {
	    next unless $seg->{TDUR} > 0;
	    push @events, {TYPE => "REF", SPKR => $spkr, EVENT => "BEG", TIME => $seg->{TBEG}};
	    push @events, {TYPE => "REF", SPKR => $spkr, EVENT => "END", TIME => $seg->{TEND}};
	}
    }
    foreach $spkr (keys %$sys_data) {
	foreach $seg (@{$sys_data->{$spkr}}) {
	    next unless $seg->{TDUR} > 0;
	    push @events, {TYPE => "SYS", SPKR => $spkr, EVENT => "BEG", TIME => $seg->{RTBEG}};
	    push @events, {TYPE => "SYS", SPKR => $spkr, EVENT => "END", TIME => $seg->{RTEND}};
	}
    }
    @events = sort {($a->{TIME} < $b->{TIME}-$epsilon  ? -1 :
		     ($a->{TIME} > $b->{TIME}+$epsilon ?  1 :
		      ($a->{EVENT} eq "END"        ? -1 : 1)))} @events;
    $evaluate = 0;
    foreach $event (@events) {
	if ($evaluate and $tbeg<$event->{TIME}) {
	    $tend = $event->{TIME};
	    push @$segments, {REF => {%ref_spkrs},
			      SYS => {%sys_spkrs},
			      TBEG => $tbeg,
			      TEND => $tend,
			      TDUR => $tend-$tbeg};
	    $tbeg = $tend;
	}
	if ($event->{TYPE} eq "UEM") {
	    $evaluate = $event->{EVENT} eq "BEG";
	    $tbeg = $event->{TIME} if $evaluate;
	}
	else {
	    $spkrs = $event->{TYPE} eq "REF" ? \%ref_spkrs : \%sys_spkrs;
	    ($event->{EVENT} eq "BEG") ? $spkrs->{$event->{SPKR}}++ : $spkrs->{$event->{SPKR}}--;
	    $spkrs->{$event->{SPKR}} <= 1 or warn
	        "WARNING:  speaker $event->{SPKR} speaking more than once at time $event->{TIME}\n";
	    delete $spkrs->{$event->{SPKR}} unless $spkrs->{$event->{SPKR}};
	}
    }
    return $segments;
}

#################################

sub sd_performance_analysis {

    my ($scores, $subtypes) = @_;
    my ($file, $chnl, $class, $kind, $ref_type, $sys_type);
    my ($xscores, %cum_scores, $count);

#accumulate statistics
    foreach $file (keys %$scores) {
	foreach $chnl (keys %{$scores->{$file}}) {
	    $xscores = $scores->{$file}{$chnl};
	    foreach $ref_type (keys %$xscores) {
		next if $ref_type eq "TYPE";
		$count = $xscores->{$ref_type};
		$cum_scores{ALL}{$ref_type} += $count;
		$cum_scores{"c=$chnl f=$file"}{$ref_type} += $xscores->{$ref_type} if $opt_a =~ /c/i and $opt_a =~ /f/i;
		$cum_scores{"c=$chnl"}{$ref_type} += $xscores->{$ref_type} if $opt_a =~ /c/i and not $opt_a =~ /f/i;
		$cum_scores{"f=$file"}{$ref_type} += $xscores->{$ref_type} if $opt_a =~ /f/i and not $opt_a =~ /c/i;
	    }
	    $xscores = $xscores->{TYPE};
	    foreach my $class ("TIME", "NSPK") {
		foreach my $kind ("REF", "SYS") {
		    foreach $ref_type (keys %{$xscores->{$class}{$kind}}) {
			$count = $xscores->{$class}{$kind}{$ref_type};
			$cum_scores{ALL}{TYPE}{$class}{$kind}{$ref_type} += $count;
			$cum_scores{"c=$chnl f=$file"}{TYPE}{$class}{$kind}{$ref_type} += $count if $opt_a =~ /c/i and $opt_a =~ /f/i;
			$cum_scores{"c=$chnl"}{TYPE}{$class}{$kind}{$ref_type} += $count if $opt_a =~ /c/i and not $opt_a =~ /f/i;
			$cum_scores{"f=$file"}{TYPE}{$class}{$kind}{$ref_type} += $count if $opt_a =~ /f/i and not $opt_a =~ /c/i;
		    }
		}
		foreach $ref_type (keys %{$xscores->{$class}{JOINT}}) {
		    foreach $sys_type (keys %{$xscores->{$class}{JOINT}{$ref_type}}) {
			$count = $xscores->{$class}{JOINT}{$ref_type}{$sys_type};
			$cum_scores{ALL}{TYPE}{$class}{JOINT}{$ref_type}{$sys_type} += $count;
			$cum_scores{"c=$chnl f=$file"}{TYPE}{$class}{JOINT}{$ref_type}{$sys_type} += $count if $opt_a =~ /c/i and $opt_a =~ /f/i;
			$cum_scores{"c=$chnl"}{TYPE}{$class}{JOINT}{$ref_type}{$sys_type} += $count if $opt_a =~ /c/i and not $opt_a =~ /f/i;
			$cum_scores{"f=$file"}{TYPE}{$class}{JOINT}{$ref_type}{$sys_type} += $count if $opt_a =~ /f/i and not $opt_a =~ /c/i;
		    }
		}
	    }
	}
    }

    foreach my $condition (sort keys %cum_scores) {
	print_sd_scores ($condition, $cum_scores{$condition}) if $condition !~ /ALL/;
    }
    print_sd_scores ("ALL", $cum_scores{ALL});
}

#################################

sub print_sd_scores {

    my ($condition, $scores) = @_;

    printf "\n*** Performance analysis for Speaker Diarization for $condition ***\n\n";

    printf "    EVAL TIME =%10.2f secs\n", $scores->{EVAL_TIME};
    printf "  EVAL SPEECH =%10.2f secs (%5.1f percent of evaluated time)\n", $scores->{EVAL_SPEECH},
        100*$scores->{EVAL_SPEECH}/$scores->{EVAL_TIME};
    printf "  SCORED TIME =%10.2f secs (%5.1f percent of evaluated time)\n",
        $scores->{SCORED_TIME}, 100*$scores->{SCORED_TIME}/$scores->{EVAL_TIME};
    printf "SCORED SPEECH =%10.2f secs (%5.1f percent of scored time)\n",
        $scores->{SCORED_SPEECH}, 100*$scores->{SCORED_SPEECH}/$scores->{SCORED_TIME};
    printf "   EVAL WORDS =%7d        \n", $scores->{EVAL_WORDS};
    printf " SCORED WORDS =%7d         (%5.1f percent of evaluated words)\n",
        $scores->{SCORED_WORDS}, 100*$scores->{SCORED_WORDS}/$scores->{EVAL_WORDS};
    print "---------------------------------------------\n";
    printf "MISSED SPEECH =%10.2f secs (%5.1f percent of scored time)\n",
        $scores->{MISSED_SPEECH}, 100*$scores->{MISSED_SPEECH}/$scores->{SCORED_TIME};
    printf "FALARM SPEECH =%10.2f secs (%5.1f percent of scored time)\n",
        $scores->{FALARM_SPEECH}, 100*$scores->{FALARM_SPEECH}/$scores->{SCORED_TIME};
    printf " MISSED WORDS =%7d         (%5.1f percent of scored words)\n",
        $scores->{MISSED_WORDS}, 100*$scores->{MISSED_WORDS}/$scores->{SCORED_WORDS};
    print "---------------------------------------------\n";
    printf "SCORED SPEAKER TIME =%10.2f secs (%5.1f percent of scored speech)\n",
        $scores->{SCORED_SPEAKER}, 100*$scores->{SCORED_SPEAKER}/$scores->{SCORED_SPEECH};
    printf "MISSED SPEAKER TIME =%10.2f secs (%5.1f percent of scored speaker time)\n",
        $scores->{MISSED_SPEAKER}, 100*$scores->{MISSED_SPEAKER}/$scores->{SCORED_SPEAKER};
    printf "FALARM SPEAKER TIME =%10.2f secs (%5.1f percent of scored speaker time)\n",
        $scores->{FALARM_SPEAKER}, 100*$scores->{FALARM_SPEAKER}/$scores->{SCORED_SPEAKER};
    printf " SPEAKER ERROR TIME =%10.2f secs (%5.1f percent of scored speaker time)\n",
        $scores->{SPEAKER_ERROR}, 100*$scores->{SPEAKER_ERROR}/$scores->{SCORED_SPEAKER};
    printf "SPEAKER ERROR WORDS =%7d         (%5.1f percent of scored speaker words)\n",
        $scores->{ERROR_WORDS}, 100*$scores->{ERROR_WORDS}/$scores->{SCORED_WORDS};
    print "---------------------------------------------\n";
#    if ($condition eq "ALL") {
#      printf " OVERALL SPEAKER DIARIZATION ERROR = %.2f percent of scored speaker time\n",
#         100*($scores->{MISSED_SPEAKER} + $scores->{FALARM_SPEAKER} + $scores->{SPEAKER_ERROR})/
#	    $scores->{SCORED_SPEAKER};
#    } else {
      printf " OVERALL SPEAKER DIARIZATION ERROR = %.2f percent of scored speaker time  %s\n",
         100*($scores->{MISSED_SPEAKER} + $scores->{FALARM_SPEAKER} + $scores->{SPEAKER_ERROR})/
    	    $scores->{SCORED_SPEAKER}, "`($condition)";
#    }
    print "---------------------------------------------\n";
    printf " Speaker type confusion matrix -- speaker weighted\n";
    summarize_speaker_type_performance ("NSPK", $scores->{TYPE}{NSPK});
    print "---------------------------------------------\n";
    printf " Speaker type confusion matrix -- time weighted\n";
    summarize_speaker_type_performance ("TIME", $scores->{TYPE}{TIME});
    print "---------------------------------------------\n";
}

#################################

sub summarize_speaker_type_performance {

    my ($class, $stats) = @_;
    my ($ref_type, $sys_type, $sys_stat);

    print "  REF\\SYS (count)      " if $class eq "NSPK";
    print "  REF\\SYS (seconds)    " if $class eq "TIME";
    foreach $sys_type ((sort keys %{$stats->{SYS}}), $miss_name) {
	printf "%-20s", $sys_type;
    }
    print "\n";

    my $ref_tot = 0;
    foreach $ref_type (keys %{$stats->{REF}}) {
	$ref_tot += $stats->{REF}{$ref_type};
    }
    
    foreach $ref_type ((sort keys %{$stats->{REF}}), $fa_name) {
	printf "%-16s", $ref_type;
	foreach $sys_type ((sort keys %{$stats->{SYS}}), $miss_name) {
	    next if $ref_type eq $fa_name and $sys_type eq $miss_name;
	    $sys_stat = $stats->{JOINT}{$ref_type}{$sys_type};
	    $sys_stat = 0 unless defined $sys_stat;
	    printf "%11d /%6.1f",   $sys_stat, min(999.9,$ref_tot ? 100*$sys_stat/$ref_tot : 9E9) if $class eq "NSPK";
	    printf "%11.2f /%6.1f", $sys_stat, min(999.9,$ref_tot ? 100*$sys_stat/$ref_tot : 9E9) if $class eq "TIME";
	    print "%";
	}
	print "\n";
    }
}

#################################

sub map_speakers {

    my ($spkr_overlap) = @_;

#compute the costs
    my $cost = {};
    foreach my $ref_spkr (keys %$spkr_overlap) {
	foreach my $sys_spkr (keys %{$spkr_overlap->{$ref_spkr}}) {
	    $cost->{$ref_spkr}{$sys_spkr} = -$spkr_overlap->{$ref_spkr}{$sys_spkr};
	}
    }

#find the mapping that maximizes the cumulative match time between ref and sys spkrs
    my $map = weighted_bipartite_graph_match ($cost);
    return $map;
}

#################################

sub inverse_speaker_map {

    my ($speaker_map) = @_;
    my ($speaker, $inverse_speaker_map);

    foreach $speaker (keys %$speaker_map) {
	$inverse_speaker_map->{$speaker_map->{$speaker}} = $speaker;
    }
    return $inverse_speaker_map;
}

#################################

sub print_speaker_map {

    my ($spkr_map, $time_overlap) = @_;
    my ($ref_spkr, $sys_spkr);

    foreach $ref_spkr (sort keys %$time_overlap) {
	$sys_spkr = $spkr_map->{$ref_spkr};
	print "'$ref_spkr' => ", defined $sys_spkr ? "'$sys_spkr'\n" : "<nil>\n";
	foreach $sys_spkr (sort keys %{$time_overlap->{$ref_spkr}}) {
	    my $time = $time_overlap->{$ref_spkr}{$sys_spkr};
	    printf "%9.2f secs matched to '$sys_spkr'\n", defined $time ? $time : 0;
	}
    }
}

#################################

sub start_speaker_map_file {
    my ($outFile) = @_;
    open (FILE, ">$outFile") || die "Error: Unable to open speaker map CSV file '$outFile' for write";
    print FILE "File,Channel,RefSpeaker,SysSpeaker,isMapped,timeOverlap\n";
    close FILE;
}

#################################

sub update_speaker_map_file {

    my ($spkr_map, $time_overlap, $file, $chnl, $outFile) = @_;

    open (FILE, ">>$outFile") || die "Error: Failed to open speaker map CSV file '$outFile' for append";
    foreach my $ref_spkr (sort keys %$time_overlap) {
	foreach my $sys_spkr (sort keys %{$time_overlap->{$ref_spkr}}) {
	    my $time = sprintf("%.4f",$time_overlap->{$ref_spkr}{$sys_spkr});
	    print FILE "$file,$chnl,$ref_spkr,$sys_spkr";
	    print FILE ",".((defined($spkr_map->{$ref_spkr}) && $sys_spkr eq $spkr_map->{$ref_spkr}) ? "mapped" : "notmapped");
	    print FILE ",$time\n";
	}
    }
    close FILE,
}

#################################

sub print_speaker_segs {

    my ($segs, $file, $chnl) = @_;
    my ($seg, @segs, $spkr, $sep);

    @segs = @$segs;
    while ($seg = shift @segs) {
	printf "beg/dur/end = %7.3f/%7.3f/%7.3f; REF = (", $seg->{TBEG}, $seg->{TDUR}, $seg->{TEND};
	print "<none>" unless defined keys %{$seg->{REF}};
	$sep = "";
	foreach $spkr (sort keys %{$seg->{REF}}) {
	    print "$sep$spkr";
	    $sep = ", ";
	}
	print "); SYS = (";
	$sep = "";
	print "<none>" unless defined keys %{$seg->{SYS}};
	foreach $spkr (sort keys %{$seg->{SYS}}) {
	    print "$sep$spkr";
	    $sep = ", ";
	}
	print "); file = $file; chnl = $chnl\n";
    }
}

#################################

sub sort_time {

    my ($token, $key) = @_;

    my $time = $token->{"R$key"};
    $time = $token->{$key} if not defined $time;
    return int(100*$time+0.5)/100
}

#################################

sub display_metadata_mapping {

    my ($file, $chnl, $ref_rttm, $sys_rttm, $ref_wds) = @_;
    my ($type, $sys_token, @events, $event, %type_cnt);
    my ($mapped, $beg_mapped, $end_mapped, $whole, $spkr_map, $sys_speaker_field);
    my %ref_tag = (NOSCORE        => "XS", NO_RT_METADATA => "NM", SEGMENT        => "SG", SPEAKER        => "SP",
		   SU             => "SU", "A/P"          => "AP", "NON-SPEECH"   => "NS", EDIT           => "ED",
		   FILLER         => "FL", IP             => "IP", CB             => "CB", "NON-LEX"      => "NL",
		   LEXEME         => "LX");
    my %sys_tag = (SPEAKER        => "SP", SU             => "SU", EDIT           => "ED", FILLER         => "FL",
		   IP             => "IP", LEXEME         => "LX");

#create a vector of rttm events
    foreach my $token (@$ref_rttm) {
	next unless defined $ref_tag{$token->{TYPE}};
	push @events, {EVENT => "BEG", TIME => sort_time ($token, "TBEG"), TYPE => $token->{TYPE}, SRC => "REF", TOKEN => $token};
	push @events, {EVENT => "END", TIME => sort_time ($token, "TEND"), TYPE => $token->{TYPE}, SRC => "REF", TOKEN => $token}
	    unless $token->{TYPE} =~ /^(IP|CB)$/;
	$token->{COUNT} = ++$type_cnt{$token->{TYPE}};
    }
    foreach my $token (@$sys_rttm) {
	next unless defined $sys_tag{$token->{TYPE}};
	push @events, {EVENT => "BEG", TIME => sort_time ($token, "TBEG"), TYPE => $token->{TYPE}, SRC => "SYS", TOKEN => $token};
	push @events, {EVENT => "END", TIME => sort_time ($token, "TEND"), TYPE => $token->{TYPE}, SRC => "SYS", TOKEN => $token}
	    unless $token->{TYPE} =~ /^(IP|CB)$/;
    }

    @events = sort sort_events @events;

    $spkr_map = inverse_speaker_map ($speaker_map{$file}{$chnl});

    print "\nChronological display of sys data aligned with ref data for file '$file', channel '$chnl'\n";
    print "----------------------- reference ----------------------- | mapped | --------------------- system output ---------------------\n";
    print "    --type-- -subtyp- -----word/spkr-----  -tbeg-  -tend- | ref_ID |     --type-- -subtyp- -----word/spkr-----  -tbeg-  -tend-\n";

    while (@events) {
        my ($token, $ref, $ref_beg, $ref_end, $sys, $sys_beg, $sys_end);
	while (@events and
	       (not $token or
		$token eq $events[0]->{TOKEN} or 
		($events[0]->{TOKEN}{MAPPTR} and
		 $token eq $events[0]->{TOKEN}{MAPPTR}))) { # collect events to display on the same line
	    $event = shift @events;
	    $token = $event->{TOKEN};
	    $event->{SRC} eq "REF" ? ($ref = $token, ($event->{EVENT} eq "BEG" ? $ref_beg : $ref_end) = 1) :
	                             ($sys = $token, ($event->{EVENT} eq "BEG" ? $sys_beg : $sys_end) = 1);
	}
	if ($ref) {
	    printf "%-3.3s %-8.8s %-8.8s %-19.19s%8s%8s | %-6.6s |",
	    (($ref->{TYPE} =~ /^(IP|CB)$/ or ($ref_beg and $ref_end)) ? "" : ($ref_beg ? "beg" : "end")),
	    $ref->{TYPE}, $ref->{SUBT},
	    $ref->{WORD} ne "<na>" ? uc $ref->{WORD} : $ref->{SPKR},
	    $ref_beg ? (sprintf "%8.2f", $ref->{TBEG}) : "",
	    $ref_end ? (sprintf "%8.2f", $ref->{TEND}) : "",
	    $ref->{MAPPTR} ? (sprintf "%s%d", $ref_tag{$ref->{TYPE}}, $ref->{COUNT}) :
		($md_subtypes{$ref->{TYPE}} ? "*Miss*" : "");
	} elsif ($sys) {
	    $ref = $sys->{MAPPTR};
	    printf "%s%8s%8s | %-6.6s |", " "x41,
	    $sys_beg ? (sprintf "%8.2f", defined $sys->{RTBEG} ? $sys->{RTBEG} : $sys->{TBEG}) : "",
	    $sys_end ? (sprintf "%8.2f", defined $sys->{RTEND} ? $sys->{RTEND} : $sys->{TEND}) : "",
	    $ref ? (sprintf "%s%d", $sys_tag{$ref->{TYPE}}, $ref->{COUNT}) :
		($md_subtypes{$sys->{TYPE}} ? "**FA**" : "");
	}
	if ($sys) {
	    $sys_speaker_field = $sys ? $sys->{SPKR} : "";
	    $sys_speaker_field .= "=>$spkr_map->{$sys->{SPKR}}" if $spkr_map->{$sys->{SPKR}};
	    printf "%3.3s %-8.8s %-8.8s %-19.19s%8s%8s",
	    (($sys->{TYPE} =~ /^(IP|CB)$/ or ($sys_beg and $sys_end)) ? "" : ($sys_beg ? "beg" : "end")),
	    $sys->{TYPE}, $sys->{SUBT},
	    $sys->{WORD} ne "<na>" ? uc $sys->{WORD} : $sys_speaker_field,
	    $sys_beg ? (sprintf "%8.2f", $sys->{TBEG}) : "",
	    $sys_end ? (sprintf "%8.2f", $sys->{TEND}) : "";
	    if ($md_subtypes{$sys->{TYPE}} and $ref = $sys->{MAPPTR}) {
		my $dw = $sys_end ?
		    ($ref->{WEND} <= $sys->{RWEND} ? 
		     delta_metadata_error_words ("END", max($ref->{WEND}, $sys->{RWBEG}-1), $sys->{RWEND}, $ref_wds) :
		     delta_metadata_error_words ("END", $ref->{WEND}, max($ref->{WBEG}-1, $sys->{RWEND}), $ref_wds)) :
		    ($ref->{WBEG} <= $sys->{RWBEG} ? 
		     delta_metadata_error_words ("BEG", $ref->{WBEG}, min(1+$ref->{WEND}, $sys->{RWBEG}), $ref_wds) :
		     delta_metadata_error_words ("BEG", min($ref->{WBEG}, 1+$sys->{RWEND}), $sys->{RWBEG}, $ref_wds));
		print " dw=$dw" if abs ($dw) > 0;
	    }
	}
	print "\n";
    }
}

#################################

sub sort_events {

    return ($a->{TIME} <=> $b->{TIME} or
	    $event_order{$a->{EVENT}} <=> $event_order{$b->{EVENT}} or
	    (($type_order{$a->{TYPE}} <=> $type_order{$b->{TYPE}})*($a->{EVENT} eq "END" ? -1 : 1)) or
	    $source_order{$a->{SRC}} <=> $source_order{$b->{SRC}});
}

#################################

sub weighted_bipartite_graph_match {
    my ($score) = @_;
    
    my $required_precision = 1E-12;
    my $INF = 1E30;
    my (@row_mate, @col_mate, @row_dec, @col_inc);
    my (@parent_row, @unchosen_row, @slack_row, @slack);
    my ($k, $l, $row, $col, @col_min, $cost, %cost);
    my $t = 0;
    
    unless (defined $score) {
	warn "input to BGM is undefined\n";
	return undef;
    }
    return {} if (keys %$score) == 0;
    
    my @rows = sort keys %{$score};
    my $miss = "miss";
    $miss .= "0" while exists $score->{$miss};
    my (@cols, %cols);
    my $min_score = $INF;
    foreach $row (@rows) {
	foreach $col (keys %{$score->{$row}}) {
	    $min_score = min($min_score,$score->{$row}{$col});
	    $cols{$col} = $col;
	}
    }
    @cols = sort keys %cols;
    my $fa = "fa";
    $fa .= "0" while exists $cols{$fa};
    my $reverse_search = @rows < @cols; # search is faster when ncols <= nrows
    foreach $row (@rows) {
	foreach $col (keys %{$score->{$row}}) {
	    ($reverse_search ? $cost{$col}{$row} : $cost{$row}{$col})
		= $score->{$row}{$col} - $min_score;
	}
    }
    push @rows, $miss;
    push @cols, $fa;
    if ($reverse_search) {
	my @xr = @rows;
	@rows = @cols;
	@cols = @xr;
    }

    my $nrows = @rows;
    my $ncols = @cols;
    my $nmax = max($nrows,$ncols);
    my $no_match_cost = -$min_score*(1+$required_precision);

    # subtract the column minimas
    for ($l=0; $l<$nmax; $l++) {
	$col_min[$l] = $no_match_cost;
	next unless $l < $ncols;
	$col = $cols[$l];
	foreach $row (keys %cost) {
	    next unless defined $cost{$row}{$col};
	    my $val = $cost{$row}{$col};
	    $col_min[$l] = $val if $val < $col_min[$l];
	}
    }
    
    # initial stage
    for ($l=0; $l<$nmax; $l++) {
	$col_inc[$l] = 0;
	$slack[$l] = $INF;
    }
    
  ROW:
    for ($k=0; $k<$nmax; $k++) {
	$row = $k < $nrows ? $rows[$k] : undef;
	my $row_min = $no_match_cost;
	for (my $l=0; $l<$ncols; $l++) {
	    my $col = $cols[$l];
	    my $val = ((defined $row and defined $cost{$row}{$col}) ? $cost{$row}{$col}: $no_match_cost) - $col_min[$l];
	    $row_min = $val if $val < $row_min;
	}
	$row_dec[$k] = $row_min;
	for ($l=0; $l<$nmax; $l++) {
	    $col = $l < $ncols ? $cols[$l]: undef;
	    $cost = ((defined $row and defined $col and defined $cost{$row}{$col}) ?
		     $cost{$row}{$col} : $no_match_cost) - $col_min[$l];
	    if ($cost==$row_min and not defined $row_mate[$l]) {
		$col_mate[$k] = $l;
		$row_mate[$l] = $k;
                # matching row $k with column $l
		next ROW;
	    }
	}
	$col_mate[$k] = -1;
	$unchosen_row[$t++] = $k;
    }
    
    goto CHECK_RESULT if $t == 0;
    
    my $s;
    my $unmatched = $t;
    # start stages to get the rest of the matching
    while (1) {
	my $q = 0;
	
	while (1) {
	    while ($q < $t) {
		# explore node q of forest; if matching can be increased, update matching
		$k = $unchosen_row[$q];
		$row = $k < $nrows ? $rows[$k] : undef;
		$s = $row_dec[$k];
		for ($l=0; $l<$nmax; $l++) {
		    if ($slack[$l]>0) {
			$col = $l < $ncols ? $cols[$l]: undef;
			$cost = ((defined $row and defined $col and defined $cost{$row}{$col}) ?
				 $cost{$row}{$col} : $no_match_cost) - $col_min[$l];
			my $del = $cost - $s + $col_inc[$l];
			if ($del < $slack[$l]) {
			    if ($del == 0) {
				goto UPDATE_MATCHING unless defined $row_mate[$l];
				$slack[$l] = 0;
				$parent_row[$l] = $k;
				$unchosen_row[$t++] = $row_mate[$l];
			    }
			    else {
				$slack[$l] = $del;
				$slack_row[$l] = $k;
			    }
			}
		    }
		}
		
		$q++;
	    }
	    
	    # introduce a new zero into the matrix by modifying row_dec and col_inc
	    # if the matching can be increased update matching
	    $s = $INF;
	    for ($l=0; $l<$nmax; $l++) {
		if ($slack[$l] and ($slack[$l]<$s)) {
		    $s = $slack[$l];
		}
	    }
	    for ($q = 0; $q<$t; $q++) {
		$row_dec[$unchosen_row[$q]] += $s;
	    }
	    
	    for ($l=0; $l<$nmax; $l++) {
		if ($slack[$l]) {
		    $slack[$l] -= $s;
		    if ($slack[$l]==0) {
			# look at a new zero and update matching with col_inc uptodate if there's a breakthrough
			$k = $slack_row[$l];
			unless (defined $row_mate[$l]) {
			    for (my $j=$l+1; $j<$nmax; $j++) {
				if ($slack[$j]==0) {
				    $col_inc[$j] += $s;
				}
			    }
			    goto UPDATE_MATCHING;
			}
			else {
			    $parent_row[$l] = $k;
			    $unchosen_row[$t++] = $row_mate[$l];
			}
		    }
		}
		else {
		    $col_inc[$l] += $s;
		}
	    }
	}
	
      UPDATE_MATCHING:  # update the matching by pairing row k with column l
	while (1) {
	    my $j = $col_mate[$k];
	    $col_mate[$k] = $l;
	    $row_mate[$l] = $k;
            # matching row $k with column $l
	    last UPDATE_MATCHING if $j < 0;
	    $k = $parent_row[$j];
	    $l = $j;
	}
	
	$unmatched--;
	goto CHECK_RESULT if $unmatched == 0;
	
	$t = 0;  # get ready for another stage
	for ($l=0; $l<$nmax; $l++) {
	    $parent_row[$l] = -1;
	    $slack[$l] = $INF;
	}
	for ($k=0; $k<$nmax; $k++) {
	    $unchosen_row[$t++] = $k if $col_mate[$k] < 0;
	}
    }  # next stage
    
  CHECK_RESULT:  # rigorously check results before handing them back
    for ($k=0; $k<$nmax; $k++) {
	$row = $k < $nrows ? $rows[$k] : undef;
	for ($l=0; $l<$nmax; $l++) {
	    $col = $l < $ncols ? $cols[$l]: undef;
	    $cost = ((defined $row and defined $col and defined $cost{$row}{$col}) ?
		     $cost{$row}{$col} : $no_match_cost) - $col_min[$l];
	    if ($cost < ($row_dec[$k] - $col_inc[$l])) {
		next unless $cost < ($row_dec[$k] - $col_inc[$l]) - $required_precision*max(abs($row_dec[$k]),abs($col_inc[$l]));
		warn "BGM: this cannot happen: cost{$row}{$col} ($cost) cannot be less than row_dec{$row} ($row_dec[$k]) - col_inc{$col} ($col_inc[$l])\n";
		return undef;
	    }
	}
    }
    
    for ($k=0; $k<$nmax; $k++) {
	$row = $k < $nrows ? $rows[$k] : undef;
	$l = $col_mate[$k];
	$col = $l < $ncols ? $cols[$l]: undef;
	$cost = ((defined $row and defined $col and defined $cost{$row}{$col}) ?
		 $cost{$row}{$col} : $no_match_cost) - $col_min[$l];
	if (($l<0) or ($cost != ($row_dec[$k] - $col_inc[$l]))) {
	    next unless $l<0 or abs($cost - ($row_dec[$k] - $col_inc[$l])) > $required_precision*max(abs($row_dec[$k]),abs($col_inc[$l]));
	    warn "BGM: every row should have a column mate: row $row doesn't, col: $col\n";
	    return undef;
	}
    }
    
    my %map;
    for ($l=0; $l<@row_mate; $l++) {
	$k = $row_mate[$l];
	$row = $k < $nrows ? $rows[$k] : undef;
	$col = $l < $ncols ? $cols[$l]: undef;
	next unless defined $row and defined $col and defined $cost{$row}{$col};
	$reverse_search ? ($map{$col} = $row) : ($map{$row} = $col);
    }
    return {%map};
}
