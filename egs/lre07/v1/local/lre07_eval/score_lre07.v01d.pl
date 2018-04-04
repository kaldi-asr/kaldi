#!/usr/bin/env perl
use warnings; #sed replacement for -w perl parameter

##############################
# history
#   v1b (Jul 30, 2007)
#        - initial release
#   v1c (Oct 11, 2007)
#        - fixed bug by changing $trial_type eq "tar" ==> $trial_type == $TARGET
#        - added code to print out the number of target and nontarget trials
#
use strict;
use Getopt::Std;
use Class::Struct;
use POSIX;

##############################
# globals
#
my $USAGE = "\n$0 [options] -t <tfile> -n <nfile>\n".
    "\n".
    "Description: This program computes the LRE07 performance measurements and generates DET points\n".
    "             as described in Sections 3.1, 3.2, and 3.4 of the LRE07 evaluation plan.\n".
    "             It creates three output files:\n".
    "                <tfile>.scr     - contains the error rates and costs for each target/non-target language pair\n".
    "                                  as well as their averages.\n".
    "                <tfile>.lng_pts - contains the miss and false alarm rates corresponding to the\n".
    "                                  DET points are language-weighted as in the new cost function.\n".
    "                <tfile>.trl_pts - contains the miss and false alarm rates corresponding to the\n".
    "                                  DET points are trial-weighted\n".
    "\n".
    "Required arguments:\n".
    "   -t <tfile>: specifies the target trials.  It contains seven fields separated by a space and in the following order:\n".
    "         <test> <tlang> <cond> <segid> <dec> <score> <slang>\n".
    "         where\n".
    "         <test>: the name of the test (general_lr | chinese_lr | mandarin_dr | english_dr | hindustani_dr | spanish_dr)\n".
    "         <tlang>:  the target language (arabic | bengali | chinese | cantonese | mandarin | mainland | taiwan |\n".
    "                   min | wu | english | american | indian | hindustani | hindi | urdu |\n".
    "                   spanish | caribbean | noncaribbean | farsi | german | japanese | korean |\n".
    "                   russia | tamil | thai | vietnamese)\n".
    "         <cond>: the nontarget language condition (closed_set | open_set)\n".
    "         <segid>: the test segment file name\n".
    "         <decision>: the decision (t | f)\n".
    "         <score>: the likelihood score\n".
    "         <slang>: the true language of the segment.  If a language has one or more subcategories,\n".
    "                  all subcategories must be included.  For example: chinese.mandarin.taiwan,\n".
    "                  chinese.cantonese, english.american, japanese, etc.\n".
    "\n".
    "   -n <nfile>: specifies the nontarget trials.  It follows the same format as <tfile>.\n".
    "\n".
    "Optional arguments:\n".
    "   -h prints this help message.\n".
    "\n";

my $DEBUG = 0;

my $TARGET = 1;
my $NONTARGET = 2;

my $MAX_NUM_FIELDS = 7;

my $OOS_LANG = "zzz"; # label for out of set language, use zzz so that it sorts last in the alphabet

# application motivated parameters
#
my $C_MISS = 1;
my $C_FA = 1;

my $P_TARGET = 0.5;
my $P_OOS = 0.2;

# data structures
#
struct Trial => {
    test => '$',
    target_language => '$',
    segment_language => '$',
    decision => '$',
    score => '$',
    type => '$'
};

##############################
# main
#
{
    # parse the commandline arguments
    #
    print "STATUS: Parsing command line arguments\n";
    my ($tar_file, $non_file) = parse_command_line_arguments();

    # read in the input files
    #
    my $tar_trials = read_trials($tar_file, $TARGET);
    my $non_trials = read_trials($non_file, $NONTARGET);

    # compute the errors (pmiss, pfa, cost)
    #
    print "STATUS: Computing errors\n";
    my %tar_lngs = ();
    my %seg_lngs = ();
    my $open_set = 0;

    my $score_file = "$tar_file.scr";
    compute_scores($score_file, $tar_trials, $non_trials, \%tar_lngs, \%seg_lngs, \$open_set);

    # compute the det points
    #
    print "STATUS: Computing language-weighted det points\n";
    my $det_file = "$tar_file.lng_pts";
    compute_language_weighted_det_points($det_file, $tar_trials, $non_trials, \%tar_lngs, \%seg_lngs, $open_set);

    print "STATUS: Computing trial-weighted det points\n";
    $det_file = "$tar_file.trl_pts";
    compute_trial_weighted_det_points($det_file, $tar_trials, $non_trials, \%tar_lngs, \%seg_lngs, $open_set);

    # done - good job
    #
    print "STATUS: Done\n";
    exit 0;
}

##############################
# subroutine
#
sub parse_command_line_arguments {

    use vars qw ( $opt_t $opt_n $opt_h );
    getopts( 't:n:h' );

    die ("$USAGE") if ( defined( $opt_h ) || ( ! defined( $opt_t) || ! defined( $opt_n ) ) );

    my $tar_file = $opt_t;
    my $non_file = $opt_n;

    if ($DEBUG > 10) {
	print "parse_command_line_arguments: target file=$tar_file\n";
	print "parse_command_line_arguments: nontarget file=$non_file\n";
    }
    return ($tar_file, $non_file);
}

sub read_trials {
    my ( $file, $type ) = @_;

    my @trials = ();
    open F, "$file" or die ("FATAL ERROR: Unable to open file '$file' reading\n");
    while (<F>) {
	if (! /^#/ && ! /^\s*$/) {
	    chomp;
	    my $line = lc( $_ );
	    my @fields = split(/\s+/, $line);

	    # make sure we got valid input
	    # we are not using the test condition and segment id fields
	    # but check them just for completeness but do not save them
	    #
	    if ( scalar( @fields ) != $MAX_NUM_FIELDS ) {
		die ("FATAL ERROR: Line '$line' does not have the required number of fields\n");
	    }
	    
	    if ($fields[0] !~ /^(general_lr|english_dr|chinese_lr|chinese_dr|mandarin_dr|hindustani_dr|spanish_dr)$/i) {
		die ("FATAL ERROR: '$fields[0]' is not a supported test in line '$line'\n");
	    }
	    
	    if ($fields[1] !~ /^(arabic|bengali|chinese|cantonese|mandarin|mainland|taiwan|min|wu|english|american|indian|hindustani|hindi|urdu|spanish|caribbean|noncaribbean|farsi|german|japanese|korean|russian|tamil|thai|vietnamese)$/i) {
		die ("FATAL ERROR: '$fields[1]' is not a supported target language in line '$line'\n");
	    }
	    
	    if ($fields[2] !~ /^(open_set|closed_set)$/i) {
		die ("FATAL ERROR: '$fields[2]' is not a supported test condition in line '$line'\n");
	    }
	    
	    if ($fields[4] !~ /^(t|f)$/i) {
		die ("FATAL ERROR: '$fields[4]' is an invalid decision value in line '$line'\n");
	    }
	    
	    if ($fields[5] !~ /[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?/) {
		die ("FATAL ERROR: '$fields[5]' must be an integer or floating point number in line '$line'\n");
	    }

	    my $trial = Trial->new();
            $trial->test($fields[0]);
            $trial->target_language($fields[1]);
            $trial->decision($fields[4]);
            $trial->score($fields[5]);
            $trial->segment_language($fields[6]);
	    $trial->type($type);

	    push( @trials, $trial);
	}
    }
    close F;

    if ($DEBUG > 10) {
	foreach my $trial (@trials) {
	    print "read_trials: " . $trial->test . " " .
		$trial->target_language . " " .
		$trial->segment_language . " " .
		$trial->decision . " " .
		$trial->score . "\n";
	}
    }
    return \@trials;
}

sub compute_scores {
    my ($score_file, $tar, $non, $tar_lngs, $seg_lngs, $open_set) = @_;

    my %miss_prob = ();    # miss probability by target language
    my $miss_prob_average = 0; # average miss probability across all target languages
    calculate_miss_rates( \%miss_prob, \$miss_prob_average,
			  $tar_lngs, $seg_lngs, $tar );

    my %fa_prob = ();
    my %fa_prob_oos = ();
    my %fa_prob_is = ();
    my %fa_prob_avg = ();

    my $fa_prob_oos_average = 0;
    my $fa_prob_is_average = 0;
    my $fa_prob_average = 0;
    calculate_fa_rates( \%fa_prob,
			\%fa_prob_oos, \$fa_prob_oos_average,
			\%fa_prob_is, \$fa_prob_is_average,
			\%fa_prob_avg, \$fa_prob_average,
			$tar_lngs, $seg_lngs, $open_set, $non );

    print_error_rates( $score_file, \%miss_prob, $miss_prob_average, \%fa_prob, \%fa_prob_oos, $fa_prob_oos_average,
		       \%fa_prob_is, $fa_prob_is_average, \%fa_prob_avg, $fa_prob_average, $tar_lngs, $seg_lngs, $non );

    my %cost = ();
    my %cost_oos = ();
    my %cost_avg = ();
    my $cost_average = 0;
    my %miss_cost = ();
    my %fa_cost = ();
    calculate_costs(\%cost, \%cost_oos, \%cost_avg, \$cost_average, \%miss_cost, \%fa_cost,
		    \%miss_prob, \%fa_prob, \%fa_prob_oos, $tar_lngs, $seg_lngs, $open_set);
    
    print_costs( $score_file, \%cost, \%cost_oos, \%cost_avg, $cost_average, $tar_lngs, $seg_lngs);
}

sub compute_language_weighted_det_points {
    my ($det, $tar, $non, $tar_lngs, $seg_lngs, $open_set) = @_;

    my $num_tar_lng = scalar( keys %$tar_lngs );
    my $num_seg_lng = scalar( keys %$seg_lngs );

    my %num_tar = ();
    my %num_non = ();
    my %num_non_oos = ();
    my %act_miss_count = ();
    my %act_fa_count = ();
    my %act_fa_count_oos = ();
    my %miss_count = ();
    my %fa_count = ();
    my %fa_count_oos = ();
    my $p_nontarget;

    if ($open_set) {
	$num_seg_lng--;
    }
    if ($num_seg_lng > 0) { # this is an extraneous check but do it anyway
	$p_nontarget = (1 - $P_TARGET - $P_OOS) / ($num_seg_lng - 1);

    } else {
	die ("FATAL ERROR: No segment language found\n");
    }

    my $total_num_tar = 0;
    my $total_num_non = 0;
    foreach my $tar_lng ( sort keys %{ $tar_lngs } ) {
	$num_tar{$tar_lng} = count_target_trials($tar, $tar_lng);
	$total_num_tar += $num_tar{$tar_lng};

	if ($num_tar{$tar_lng} == 0) {
	    die ("FATAL ERROR: No target trials for target language '$tar_lng'\n");
	}
	$act_miss_count{$tar_lng} = 0;
	$miss_count{$tar_lng} = 0;
	foreach my $seg_lng ( sort keys %{ $seg_lngs } ) {
	    if ($tar_lng ne $seg_lng && $seg_lng ne $OOS_LANG) {
		$num_non{$tar_lng}{$seg_lng} = count_nontarget_trials($non, $tar_lng, $seg_lng);
		$total_num_non += $num_non{$tar_lng}{$seg_lng};
		if ($num_non{$tar_lng}{$seg_lng} == 0) {
		    die ("FATAL ERROR: No nontarget trials for target language '$tar_lng' and segment language '$seg_lng' 1\n");
		}
		$act_fa_count{$tar_lng}{$seg_lng} = 0;
		$fa_count{$tar_lng}{$seg_lng} = $num_non{$tar_lng}{$seg_lng};
	    }
	    
	    if ($open_set) {
		$num_non_oos{$tar_lng} = count_nontarget_trials($non, $tar_lng, $OOS_LANG);
		if ($num_non_oos{$tar_lng} == 0) {
		    die ("FATAL ERROR: No out-of-set nontarget trials for target language '$tar_lng'\n");
		}
		$act_fa_count_oos{$tar_lng} = 0;
		$fa_count_oos{$tar_lng} = $num_non_oos{$tar_lng};
	    }
	}
    }

    # sort the trials by increasing likelihood score.  If same score
    # the target trial should preceed the nontarget trial
    #
    my $merged_trials = merge_trials ($tar, $non);
    my @sorted_merged_trials = sort numerically (@$merged_trials);

    if ($DEBUG > 1) {
	foreach my $trial ( @sorted_merged_trials ) {
	    print "compute_language_weighted_det_points (sorted trials): " . $trial->test . " " .
		$trial->target_language . " " .
		$trial->segment_language . " " .
		$trial->decision . " " .
		$trial->score . "\n";
	}
    }

    my @first_det_point = ();
    my @det_points = ();

    my $min_cost = FLT_MAX;
    my $min_fa_prob;
    my $min_miss_prob;
    my $fa_prob = 1;
    my $miss_prob = 0;

    push (@det_points, [ $fa_prob, $miss_prob ]);

    foreach my $trial ( @sorted_merged_trials ) {
	my $tar_lng = $trial->target_language;

	if ( $trial->type == 1 ) {
	    # target trial
	    #
	    if ( $trial->decision eq "f" ) {
		$act_miss_count{$tar_lng}++;
	    }
	    $miss_count{$tar_lng}++;
	    $miss_prob = 0;
	    for my $tr_lng ( sort keys %$tar_lngs ) {
		$miss_prob += $miss_count{$tr_lng} / ($num_tar{$tr_lng}*$num_tar_lng) if $num_tar{$tr_lng} > 0;
	    }

	} else {
	    # nontarget trial
	    #
	    my $seg_lng = normalize_language_name( $trial->segment_language, $trial->test );
	    if ($seg_lng eq $OOS_LANG) {
		$fa_count_oos{$tar_lng}--;
		$fa_prob = 0;
		for my $tr_lng ( sort keys %$tar_lngs ) {
		    $fa_prob += ($fa_count_oos{$tr_lng}/($num_non_oos{$tr_lng}*$num_tar_lng)) * ($P_OOS/(1-$P_TARGET)) if $num_non_oos{$tr_lng} > 0;
		    for my $sg_lng ( sort keys %$seg_lngs ) {
			    if ($sg_lng ne $tr_lng && $sg_lng ne $OOS_LANG) {
			    $fa_prob += ($fa_count{$tr_lng}{$sg_lng}/($num_non{$tr_lng}{$sg_lng}*($num_seg_lng-1)*$num_tar_lng)) * ((1-$P_TARGET-$P_OOS)/(1-$P_TARGET)) if $num_non{$tr_lng}{$sg_lng} > 0;
			}
		    }
		}
	    } else {
		if ($open_set) {
		    $fa_count{$tar_lng}{$seg_lng}--;
		    $fa_prob = 0;
		    for my $tr_lng ( sort keys %$tar_lngs ) {
			$fa_prob += ($fa_count_oos{$tr_lng}/($num_non_oos{$tr_lng}*$num_tar_lng)) * ($P_OOS/(1-$P_TARGET)) if $num_non_oos{$tr_lng} > 0;
			for my $sg_lng ( sort keys %$seg_lngs ) {
			    if ($sg_lng ne $tr_lng && $sg_lng ne $OOS_LANG) {
				$fa_prob += ($fa_count{$tr_lng}{$sg_lng}/($num_non{$tr_lng}{$sg_lng}*($num_seg_lng-1)*$num_tar_lng)) * ((1-$P_TARGET-$P_OOS)/(1-$P_TARGET)) if $num_non{$tr_lng}{$sg_lng} > 0;
			    }
			}
		    }
		} else {
		    $fa_count{$tar_lng}{$seg_lng}--;
		    $fa_prob = 0;
		    for my $tr_lng ( sort keys %$tar_lngs ) {
			for my $sg_lng ( sort keys %$seg_lngs ) {
			    if ($sg_lng ne $tr_lng && $sg_lng ne $OOS_LANG) {
				$fa_prob += $fa_count{$tr_lng}{$sg_lng}/($num_non{$tr_lng}{$sg_lng}*($num_seg_lng-1)*$num_tar_lng);
			    }
			}
		    }
		}
	    }

	    if ( $trial->decision eq "t" ) {
		if ( $seg_lng eq $OOS_LANG ) {
		    $act_fa_count_oos{$tar_lng}++;
		} else {
		    $act_fa_count{$tar_lng}{$seg_lng}++;
		}
	    }
	}

	push (@det_points, [ $fa_prob, $miss_prob ]);
	my $cost = ($C_FA * $fa_prob * (1 - $P_TARGET)) + ($C_MISS * $miss_prob * $P_TARGET);
	if ($cost < $min_cost) {
	    $min_cost = $cost;
	    $min_fa_prob = $fa_prob;
	    $min_miss_prob = $miss_prob;
	}
    }

    if ( $DEBUG > 8 ) {
	foreach my $pair (@det_points) {
	    printf ("compute_language_weighted_det_points: %.8f %.8f\n", $pair->[0], $pair->[1]);
	}
    }

    # print out the det points
    #
    open O, ">$det" or die ("FATAL ERROR: Unable to open '$det' for writing\n");
    foreach my $pair (@det_points) {
	printf O ("%.8f %.8f\n", $pair->[0], $pair->[1]);
    }
    close O;

    # print out the min det point
    #
    open O, ">>$det" or die ("FATAL ERROR: Unable to open '$det' for writing\n");
    printf O ("min_det %.8f %.8f\n", $min_fa_prob, $min_miss_prob);
    printf O ("min_cost %.8f %.8f\n", $C_FA*$min_fa_prob*(1-$P_TARGET), $C_MISS*$min_miss_prob*$P_TARGET);
    close O;

    # calculate actual det point
    #
    my %act_miss_prob = ();
    my %act_fa_prob = ();
    my %act_fa_prob_oos = ();
    my %act_fa_prob_is = ();
    my %act_fa_prob_avg = ();
    my $act_miss_prob_average = 0;
    my $act_fa_prob_average = 0;

    foreach my $tar_lng ( sort keys %{ $tar_lngs } ) {
	$act_miss_prob{$tar_lng} = $act_miss_count{$tar_lng} / $num_tar{$tar_lng};
	$act_miss_prob_average += $act_miss_prob{$tar_lng};
	$act_fa_prob_is{$tar_lng} = 0;

	foreach my $seg_lng ( sort keys %{ $seg_lngs } ) {
	    if ($tar_lng ne $seg_lng && $seg_lng ne $OOS_LANG) {
		$act_fa_prob{$tar_lng}{$seg_lng} = $act_fa_count{$tar_lng}{$seg_lng} / $num_non{$tar_lng}{$seg_lng};
		$act_fa_prob_is{$tar_lng} += $act_fa_prob{$tar_lng}{$seg_lng};
	    }
	}

	if ($open_set) {
	    $act_fa_prob_oos{$tar_lng} = $act_fa_count_oos{$tar_lng} / $num_non_oos{$tar_lng};
	    $act_fa_prob_avg{$tar_lng} = (($p_nontarget * $act_fa_prob_is{$tar_lng}) + ($P_OOS * $act_fa_prob_oos{$tar_lng})) / (1-$P_TARGET);
	} else {
	    $act_fa_prob_avg{$tar_lng} = $act_fa_prob_is{$tar_lng} / ($num_seg_lng - 1);
	}
	$act_fa_prob_average += $act_fa_prob_avg{$tar_lng};
    }

    $act_fa_prob_average /= $num_tar_lng;
    $act_miss_prob_average /= $num_tar_lng;

    # print out the act det point
    #
    open O, ">>$det" or die ("FATAL ERROR: Unable to open '$det' for writing\n");
    printf O ("act_det %.8f %.8f\n", $act_fa_prob_average, $act_miss_prob_average);
    printf O ("act_cost %.8f %.8f\n", $C_FA*$act_fa_prob_average*(1-$P_TARGET), $C_MISS*$act_miss_prob_average*$P_TARGET);
    printf O ("num_tar %d\n", $total_num_tar);
    printf O ("num_non %d\n", $total_num_non);
    close O;
}

sub compute_trial_weighted_det_points {
    my ($det, $tar, $non, $tar_lngs, $seg_lngs, $open_set) = @_;

    my $num_tar_trials = scalar( @$tar );
    my $num_non_trials = scalar( @$non );

    # sort the trials by increasing likelihood score.  If same score
    # the target trial should preceed the nontarget trial
    #
    my $merged_trials = merge_trials ($tar, $non);
    my @sorted_merged_trials = sort numerically (@$merged_trials);

    my @det_points = ();

    my $min_cost = FLT_MAX;
    my $min_fa_count;
    my $min_miss_count;
    my $fa_count = $num_non_trials;
    my $miss_count = 0;
    my $act_miss_count = 0;
    my $act_fa_count = 0;

    push (@det_points, [ $fa_count/$num_non_trials, $miss_count/$num_tar_trials ]);

    foreach my $trial ( @sorted_merged_trials ) {
	my $trial_type = $trial->type;
	my $dec = $trial->decision;

	if ( $trial_type == $TARGET ) {
	    # target trial
	    #

	    if ( $dec eq "f" ) {
		$act_miss_count++;
	    }
	    $miss_count++;

	} else {
	    # nontarget trial
	    #
	    if ( $dec eq "t" ) {
		$act_fa_count++;
	    }
	    $fa_count--;
	}

	push (@det_points, [ $fa_count/$num_non_trials, $miss_count/$num_tar_trials ]);
	my $cost = ($C_FA * $fa_count/$num_non_trials * (1 - $P_TARGET)) + ($C_MISS * $miss_count/$num_tar_trials * $P_TARGET);
	if ($cost < $min_cost) {
	    $min_cost = $cost;
	    $min_fa_count = $fa_count;
	    $min_miss_count = $miss_count;
	}
    }

    # print out the det points
    #
    open O, ">$det" or die ("FATAL ERROR: Unable to open '$det' for writing\n");
    foreach my $pair (@det_points) {
	printf O ("%.8f %.8f\n", $pair->[0], $pair->[1]);
    }
    close O;

    # print out the min det point
    #
    open O, ">>$det" or die ("FATAL ERROR: Unable to open '$det' for writing\n");
    printf O ("min_det %.8f %.8f\n", $min_fa_count/$num_non_trials, $min_miss_count/$num_tar_trials);
    printf O ("min_cost %.8f %.8f\n", $C_FA*$min_fa_count/$num_non_trials*(1-$P_TARGET), $C_MISS*$min_miss_count/$num_tar_trials*$P_TARGET);

    # print out the act det point
    #
    open O, ">>$det" or die ("FATAL ERROR: Unable to open '$det' for writing\n");
    printf O ("act_det %.8f %.8f\n", $act_fa_count/$num_non_trials, $act_miss_count/$num_tar_trials);
    printf O ("act_cost %.8f %.8f\n", $C_FA*$act_fa_count/$num_non_trials*(1-$P_TARGET), $C_MISS*$act_miss_count/$num_tar_trials*$P_TARGET);
    printf O ("num_tar %d\n", $num_tar_trials);
    printf O ("num_non %d\n", $num_non_trials);

    close O;
}

sub normalize_language_name {
    my ($seg_lng, $test) = @_;

    if (is_out_of_set_language($seg_lng, $test)) {
	$seg_lng = $OOS_LANG;
	
    } else {
	my ($a, $b, $c) = split(/\./, $seg_lng, 3);
	
	if ($test eq "general_lr") {
	    $seg_lng = $a;
	    
	} elsif ($test eq "english_dr" ||
		 $test eq "hindustani_dr" ||
		 $test eq "spanish_dr") {
	    if (defined($b)) {
		$seg_lng = $b;
	    }
	    
	} elsif ($test eq "chinese_lr") {
	    if (defined($b)) {
		$seg_lng = $b;
	    }
	    
	} elsif ($test eq "mandarin_dr") {
	    if (defined($c)) {
		$seg_lng = $c;
	    }
	}
    }
    
    return $seg_lng;
}

sub is_out_of_set_language {
    my ($lng, $test) = @_;

    if ( $test eq "general_lr" ) {
	if ($lng =~ /(arabic|bengali|farsi|german|japanese|korean|russian|tamil|thai|vietnamese|chinese|english|hindustani|spanish)/i) {
	    return 0;
	}

    } elsif ( $test eq "chinese_lr" ) {
	if ( $lng =~ /(cantonese|mandarin|min|wu)/i )  {
	    return 0;
	}

    } elsif ( $test eq "english_dr" ) {
	if ( $lng =~ /(american|indian)/i ) {
	    return 0;
	}

    } elsif ( $test eq "hindustani_dr" ) {
	if ( $lng =~ /(hindi|urdu)/i ) {
	    return 0;
	}

    } elsif ( $test eq "mandarin_dr" ) {
	if ( $lng =~ /(mainland|taiwan)/i ) {
	    return 0;
	}

    } elsif ( $test eq "spanish_dr" ) {
	if ( $lng =~ /(caribbean|noncaribbean)/i ) {
	    return 0;
	}
    }
    return 1;
}

sub calculate_miss_rates {
    my ($miss_prob, $miss_prob_average, $tar_lngs, $seg_lngs, $tar) = @_;

    my %miss_count = ();  # miss count by target language
    my %total_count = (); # total count of target trials by target language

    find_target_languages($tar_lngs, $tar);

    # initialize to avoid uninitialized value errors
    #
    for my $tar_lng ( sort keys %$tar_lngs ) {
	$miss_count{$tar_lng} = 0;
	$total_count{$tar_lng} = 0;
	$miss_prob->{$tar_lng} = 0;
	$seg_lngs->{$tar_lng} = 0;
    }
    $$miss_prob_average = 0;

    # count the misses
    #
    foreach my $trial ( @{ $tar } ) {
	my $tar_lng = $trial->target_language;
	my $seg_lng = normalize_language_name( $trial->segment_language, $trial->test );
	my $dec = $trial->decision;

	if ($dec eq "f") {
	    $miss_count{$tar_lng}++;
	}
	$total_count{$tar_lng}++;
    }

    # calculate the miss rates
    #
    foreach my $tar_lng ( sort keys %$tar_lngs ) {
	$miss_prob->{$tar_lng} = $miss_count{$tar_lng} / $total_count{$tar_lng};
	$$miss_prob_average += $miss_prob->{$tar_lng};
    }

    # calculate the average miss rate
    #
    my $num_tar_lng = scalar( keys %$tar_lngs );

    if ($num_tar_lng != 0) {
	$$miss_prob_average /= $num_tar_lng;
    } else {
	die ("FATAL ERROR: No target language found 1\n");
    }
}

sub calculate_fa_rates {

    my ($fa_prob,
	$fa_prob_oos, $fa_prob_oos_average,
	$fa_prob_is, $fa_prob_is_average,
	$fa_prob_avg, $fa_prob_average,
	$tar_lngs, $seg_lngs, $open_set, $non) = @_;

    my %fa_count = ();    # false alarm count by target and segment language where segment language is not unknown
    my %total_count = (); # total count of nontarget trials by target and segment language

    my %fa_count_oos = ();      # false alarm count by target language where segment language is unknown
    my %total_count_oos = ();   # total count of nontarget trials by target language

    find_segment_languages($seg_lngs, $non);

    # initialize to avoid uninitialized value errors
    #
    for my $tar_lng ( sort keys %$tar_lngs ) {
	for my $seg_lng ( sort keys %$seg_lngs ) {
	    $fa_count{$tar_lng}{$seg_lng} = 0;
	    $total_count{$tar_lng}{$seg_lng} = 0;
	    $fa_prob->{$tar_lng}{$seg_lng} = 0;
	}
	$fa_count_oos{$tar_lng} = 0;
	$total_count_oos{$tar_lng} = 0;
	$fa_prob_oos->{$tar_lng} = 0;
	$fa_prob_is->{$tar_lng} = 0;
	$fa_prob_avg->{$tar_lng} = 0;
    }

    $$fa_prob_is_average = 0;
    $$fa_prob_oos_average = 0;
    $$fa_prob_average = 0;
    $$open_set = 0; # initially closed-set

    # count the false alarms
    #
    foreach my $trial ( @{ $non } ) {
	my $tar_lng = $trial->target_language;

	# only deal with the nontarget trials whose target languages are among those for target trials
	#
	if ( exists( $tar_lngs->{$tar_lng} ) ) {
	    my $seg_lng = normalize_language_name( $trial->segment_language, $trial->test );
	    my $dec = $trial->decision;
	    
	    if ($seg_lng eq $OOS_LANG) {
		# open set
		#
		$$open_set = 1;
		
		if ( $dec eq "t" ) {
		    $fa_count_oos{$tar_lng}++;
		}
		$total_count_oos{$tar_lng}++;

	    } else {
		# closed set
		#
		if ($tar_lng eq $seg_lng) { # is this check extraneous since if it's nontarget trial tar != seg
		    die ("FATAL ERROR: Target '$tar_lng' and segment '$seg_lng' languages must be different\n");
		}
		
		if ( $dec eq "t" ) {
		    $fa_count{$tar_lng}{$seg_lng}++;
		}
		$total_count{$tar_lng}{$seg_lng}++;
	    }
	}
    }

    # calculate the false alarm rates
    #
    foreach my $tar_lng ( sort keys %$tar_lngs ) {
	foreach my $seg_lng ( sort keys %$seg_lngs ) {
	    if ( $tar_lng ne $seg_lng && $seg_lng ne $OOS_LANG ) {
		if ( $total_count{$tar_lng}{$seg_lng} == 0 ) {
		    die ("FATAL ERROR: No nontarget trials for target language '$tar_lng' and segment language '$seg_lng' 2\n");
		}
		
		$fa_prob->{$tar_lng}{$seg_lng} = $fa_count{$tar_lng}{$seg_lng} / $total_count{$tar_lng}{$seg_lng};
		
		$fa_prob_is->{$tar_lng} += $fa_prob->{$tar_lng}{$seg_lng};
	    }
	}

	my $num_seg_lng = scalar( keys %$seg_lngs );

	if ($$open_set) {
	    $num_seg_lng--;
	}

	if ( $num_seg_lng != 0 ) {
	    $fa_prob_is->{$tar_lng} /= ( $num_seg_lng - 1 );
	    $$fa_prob_is_average += $fa_prob_is->{$tar_lng};

	} else {
	    die ("FATAL ERROR: No target language found 2\n");
	}

	if ( $$open_set ) {
	    if ( $total_count_oos{$tar_lng} == 0 ) {
		die ("FATAL ERROR: No out of set nontarget trials for target language '$tar_lng'\n");
	    }
	    
	    $fa_prob_oos->{$tar_lng} = $fa_count_oos{$tar_lng} / $total_count_oos{$tar_lng};
	    $fa_prob_avg->{$tar_lng} = ( ( $fa_prob_is->{$tar_lng} * (1 - $P_TARGET - $P_OOS) ) + 
					 ( $fa_prob_oos->{$tar_lng} * $P_OOS ) ) / ( 1 - $P_TARGET);

	    $$fa_prob_oos_average += $fa_prob_oos->{$tar_lng};

	} else {
	    $fa_prob_avg->{$tar_lng} = $fa_prob_is->{$tar_lng};
	}

	$$fa_prob_average += $fa_prob_avg->{$tar_lng};
    }

    # calculate the average fa rate
    #
    my $num_tar_lng = scalar( keys %$tar_lngs );

    if ($num_tar_lng != 0) {
	$$fa_prob_is_average /= $num_tar_lng;
	$$fa_prob_oos_average /= $num_tar_lng;
	$$fa_prob_average /= $num_tar_lng;

    } else {
	die ("FATAL ERROR: No target language found 3\n");
    }
}

sub print_error_rates {

    my ($file, $miss_prob, $miss_prob_average, $fa_prob, $fa_prob_oos, $fa_prob_oos_average,
	$fa_prob_is, $fa_prob_is_average, $fa_prob_avg, $fa_prob_average, $tar_lngs, $seg_lngs) = @_;

    open O, ">$file" or die ("FATAL ERROR: Unable to open file '$file' for writing\n");

    print O "########################### LEGEND:\n";
    for my $tar_lng (sort keys %{ $tar_lngs } ) {
	printf O "# %s - $tar_lng\n",  uc( substr( $tar_lng, 0, 3 ) );
    }
    print O "# OOS - out-of-set languages, if any\n";
    print O "#\n";

    print O "# Lt - Target languages\n";
    print O "# Ln - Segment languages\n";
    print O "# Lo - Out-of-set languages including both 'unknown' languages and 'known' but is not in the language set for a given test\n";
    print O "#\n";
    print O "# Pmiss(Lt) - Miss prob for the corresponding target language Lt\n";
    print O "# Avg Pfa(Lt) - Avg fa prob for the corresponding target language Lt across all the closed set segment languages\n";
    print O "# Avg Pfa(Lt+Lo) - Avg fa prob for the corresponding target language Lt across all segment languages\n";
    print O "#\n";
    print O "# Avg Pmiss - Avg miss across all the target languages\n";
    print O "# Avg Pfa(Lo) - Avg fa across all the target languages when segment language is out-of-set\n";
    print O "# Avg Pfa(not Lo) - Avg fa across all the target languages when segment language is not out-of-set\n";
    print O "# Avg Pfa - Avg fa across all the target languages\n";
    print O "#\n";
    print O "# Avg Cost(Lt) - Avg cost for the corresponding target language Lt across all segment languages\n";
    print O "# Avg Cost - Avg cost across all the target languages\n";
    print O "\n";

    # print out the error rates
    #
    print O "########################### ERROR RATES: Pfa(Lt,Ln)\n";
    my $col = scalar( keys %{ $tar_lngs } );
    my $mid_col = int( $col / 2 );
    printf O "%-20s | ", " ";
    for (my $i=1; $i<=$col; $i++) {
	if ($i == $mid_col) {
	    printf O "%-7s   ", "Target";
	} elsif ($i == $mid_col+1) {
	    printf O "%-7s  ", "Language";
	} elsif ($i == $mid_col+2) {
	    printf O "%-7s   ", "Lt";
	} elsif ($i == $col) {
	    printf O "%-7s | ", " ";
	} else {
	    printf O "%-7s   ", " ";
	}
    }
    print O "\n";
    printf O "%-20s | ", "Segment Language Ln";
    for my $tar_lng (sort keys %{ $tar_lngs } ) {
	printf O "%-7s | ", uc( substr( $tar_lng, 0, 3 ) );
    }
    print O "\n";

    my $open_set = 0;
    for my $seg_lng (sort keys %{ $seg_lngs } ) {
	if ($seg_lng eq $OOS_LANG) {
	    printf O "%-20s | ", "OOS";
	    $open_set = 1;
	} else {
	    printf O "%-20s | ", uc( substr( $seg_lng, 0, 3 ) );
	}

	for my $tar_lng (sort keys %{ $tar_lngs } ) {
	    if ( $tar_lng eq $seg_lng ) {
		printf O "%-7s | ", "--";

	    } else {
		if ($seg_lng ne $OOS_LANG) {
		    printf O "%-7.4f | ", $fa_prob->{$tar_lng}{$seg_lng};
		} else {
		    printf O "%-7.4f | ", $fa_prob_oos->{$tar_lng};
		}
	    }
	}
	print O "\n";
    }
    print O "------------------";
    for my $tar_lng (sort keys %{ $tar_lngs } ) {
	print O "----------";
    }
    print O "\n";

    printf O "%-20s | ", "Pmiss(Lt)";
    for my $tar_lng (sort keys %{ $tar_lngs } ) {
	printf O "%-7.4f | ", $miss_prob->{$tar_lng};
    }
    print O "\n";

    printf O "%-20s | ", "Avg Pfa(Lt)";
    for my $tar_lng (sort keys %{ $tar_lngs } ) {
	printf O "%-7.4f | ", $fa_prob_is->{$tar_lng};
    }
    print O "\n";

    printf O "%-20s | ", "Avg Pfa(Lt+Lo)";
    for my $tar_lng (sort keys %{ $tar_lngs } ) {
	printf O "%-7.4f | ", $fa_prob_avg->{$tar_lng};
    }
    print O "\n";

    print O "\n";

    printf O "%-20s = %-7.4f\n", "Avg Pmiss", $miss_prob_average;
    printf O "%-20s = %-7.4f\n", "Avg Pfa (not Lo)", $fa_prob_is_average;
    if ($open_set) {
	printf O "%-20s = %-7.4f\n", "Avg Pfa (Lo)", $fa_prob_oos_average;
    }
    printf O "%-20s = %-7.4f\n", "Avg Pfa", $fa_prob_average;
}

sub print_costs {

    my ($file, $cost, $cost_oos, $cost_avg, $cost_average, $tar_lngs, $seg_lngs) = @_;

    open O, ">>$file" or die ("FATAL ERROR: Unable to open file '$file' for appending\n");

    print O "\n";

    # print out the costs
    #
    print O "########################### COSTS: C(Lt,Ln)\n";
    my $col = scalar( keys %{ $tar_lngs } );
    my $mid_col = int( $col / 2 );
    printf O "%-20s | ", " ";
    for (my $i=1; $i<=$col; $i++) {
	if ($i == $mid_col) {
	    printf O "%-7s   ", "Target";
	} elsif ($i == $mid_col+1) {
	    printf O "%-7s  ", "Language";
	} elsif ($i == $mid_col+2) {
	    printf O "%-7s   ", "Lt";
	} elsif ($i == $col) {
	    printf O "%-7s | ", " ";
	} else {
	    printf O "%-7s   ", " ";
	}
    }
    print O "\n";
    printf O "%-20s | ", "Segment Language Ln";
    for my $tar_lng (sort keys %{ $tar_lngs } ) {
	printf O "%-7s | ", uc( substr( $tar_lng, 0, 3 ) );
    }
    print O "\n";
    
    for my $seg_lng (sort keys %{ $seg_lngs } ) {
	if ($seg_lng eq $OOS_LANG) {
	    printf O "%-20s | ", "OOS";
	} else {
	    printf O "%-20s | ", uc( substr( $seg_lng, 0, 3 ) );
	}

	for my $tar_lng (sort keys %{ $tar_lngs } ) {
	    if ( $tar_lng eq $seg_lng ) {
		printf O "%-7s | ", "--";
	    } else {
		if ($seg_lng ne $OOS_LANG) {
		    printf O "%-7.4f | ", $cost->{$tar_lng}{$seg_lng};
		} else {
		    printf O "%-7.4f | ", $cost_oos->{$tar_lng};
		}
	    }
	}
	print O "\n";
    }

    print O "------------------";
    for my $tar_lng (sort keys %{ $tar_lngs } ) {
	print O "----------";
    }
    print O "\n";

    printf O "%-20s | ", "Avg Cost(Lt)";
    for my $tar_lng (sort keys %{ $tar_lngs } ) {
	printf O "%-7.4f | ", $cost_avg->{$tar_lng};
    }
    print O "\n";

    print O "\n";

    printf O "%-20s = %-7.4f\n", "Avg Cost", $cost_average;

}

sub calculate_costs {

    my ($cost, $cost_oos, $cost_avg, $cost_average, $miss_cost, $fa_cost,
	$miss_prob, $fa_prob, $fa_prob_oos, $tar_lngs, $seg_lngs, $open_set) = @_;

    $$cost_average = 0;

    foreach my $tar_lng ( sort keys %{ $tar_lngs } ) {
	$miss_cost->{$tar_lng} = $C_MISS * $P_TARGET * $miss_prob->{$tar_lng};

	$fa_cost->{$tar_lng} = 0;
	foreach my $seg_lng ( sort keys %{ $seg_lngs } ) {

	    if ( $tar_lng ne $seg_lng && $seg_lng ne $OOS_LANG ) {

		$fa_cost->{$tar_lng} += $C_FA * (1 - $P_TARGET) * $fa_prob->{$tar_lng}{$seg_lng};
		$cost->{$tar_lng}{$seg_lng} = $miss_cost->{$tar_lng} + $C_FA * (1 - $P_TARGET) * $fa_prob->{$tar_lng}{$seg_lng};
	    }
	}

	my $num_seg_lng = scalar( keys %{ $seg_lngs } );

	if ($$open_set) {
	    $num_seg_lng--;
	}
	if ( $num_seg_lng > 0 ) {
	    $fa_cost->{$tar_lng} /= ( $num_seg_lng - 1 );

	} else {
	    die ("FATAL ERROR: No segment language found\n");
	}

	if ($$open_set) {
	    $cost_oos->{$tar_lng} = $miss_cost->{$tar_lng} + ( $C_FA * (1 - $P_TARGET) * $fa_prob_oos->{$tar_lng} );
	    $fa_cost->{$tar_lng} = ( $fa_cost->{$tar_lng} * (1 - $P_TARGET - $P_OOS) ) / (1 - $P_TARGET)  + 
		( $C_FA * $P_OOS * $fa_prob_oos->{$tar_lng} );
	}
	$cost_avg->{$tar_lng} = $miss_cost->{$tar_lng} + $fa_cost->{$tar_lng};
	$$cost_average += $cost_avg->{$tar_lng};
    }

    my $num_tar_lng = scalar( keys %{ $tar_lngs } );

    if ($num_tar_lng != 0) {
	$$cost_average /= $num_tar_lng;

    } else {
	die ("FATAL ERROR: No target language found 4\n");
    }
}

sub count_target_trials {
    my ($trials, $tar_lang) = @_;

    my $count = 0;

    foreach my $trial ( @{ $trials } ) {
	if ( $trial->target_language eq $tar_lang) {
	    $count++;
	}
    }

    return $count;
}

sub count_nontarget_trials {
    my ($trials, $tar_lang, $seg_lang) = @_;

    my $count = 0;

    foreach my $trial ( @{ $trials } ) {
	my $norm_seg_lng = normalize_language_name( $trial->segment_language, $trial->test );
	if ( $trial->target_language eq $tar_lang && $norm_seg_lng eq $seg_lang) {
	    $count++;
	}
    }

    return $count;
}

sub find_target_languages {
    my ($tar_lngs, $trials) = @_;

    foreach my $trial ( @{ $trials } ) {
	my $tar_lng = $trial->target_language;
	$tar_lngs->{$tar_lng} = 0;
    }

    if ($DEBUG > 1) {
	foreach my $tar_lng ( sort keys %$tar_lngs ) {
	    print "find_target_language: $tar_lng\n";
	}
    }
}

sub find_segment_languages {
    my ($seg_lngs, $trials) = @_;

    foreach my $trial ( @{ $trials } ) {
	my $seg_lng = normalize_language_name( $trial->segment_language, $trial->test );
	$seg_lngs->{$seg_lng} = 0;
    }

    if ($DEBUG > 1) {
	foreach my $seg_lng ( sort keys %$seg_lngs ) {
	    print "find_segment_language: $seg_lng\n";
	}
    }
}

sub merge_trials {
    my ($tar, $non) = @_;

    my @merged_trials = ();
    foreach my $trial ( @$tar ) {
	push ( @merged_trials, $trial );
    }

    foreach my $trial ( @$non ) {
	push ( @merged_trials, $trial );
    }

    if ($DEBUG > 1) {
	foreach my $trial ( @merged_trials ) {
	    print "merge_trials: " . $trial->test . " " .
		$trial->target_language . " " .
		$trial->segment_language . " " .
		$trial->decision . " " .
		$trial->score . "\n";
	}
    }
    return \@merged_trials;
}

sub numerically {

    if ($a->score > $b->score) {
	return 1;

    } elsif ($a->score < $b->score) {
	return -1;

    } else {
	if ($a->type == 2) {
	    return 1;
	} elsif ($a->type == 1) {
	    return -1;
	}
    }
}
