eval '(exit $?0)' && eval 'exec perl -w -S $0 ${1+"$@"}' && eval 'exec perl -w -S $0 $argv:q'
  if 0;
# The above two evil-looking lines enable the perl interpreter to be found on
# the PATH when this script is executed as a Unix command.

###################################################################
# Copyright 2008 by BBN Technologies Corp.     All Rights Reserved
###################################################################

use strict;

use File::Basename;

my $usage =
    "Usage: $0 [options] reference hypothesis [output-prefix]\n".
    "Options:\n".
    "   --ref_format <trn|stm|kaldi> reference file format (default is kaldi)\n".
    "   --hyp_format <trn|ctm|kaldi> hypothesis file format (default is kaldi)\n".    "                             trn denotes SNOR format transcript:\n".
    "                             line ends with unique ID in parentheses,\n".
    "                             no other parentheses allowed.\n";

# Set default formats

my %ref_format_values = map { $_ => 1 } ("trn", "stm", "kaldi" );
my %hyp_format_values = map { $_ => 1 } ("trn", "ctm", "kaldi" );

my $ref_format = "kaldi";
my $hyp_format = "kaldi";
my $SPEAKER_SIDE_DELIM = "-";
my $require_all_in_ref = 0;

use Getopt::Long;

if ( ! GetOptions( "ref_format=s" => \$ref_format,
                   "hyp_format=s" => \$hyp_format,
                   "use_all_ref=i" => \$require_all_in_ref) ) {
    die $usage;
}

if ( ! $ref_format_values{$ref_format} ) {
    die "Invalid format for reference file: $ref_format\n$usage";
}

if ( ! $hyp_format_values{$hyp_format} ) {
    die "Invalid format for hypothesis file: $hyp_format\n$usage";
}

die $usage
    if ( ( @ARGV > 3 ) || ( @ARGV < 2 ) );

my $ref = $ARGV[0];
my $hyp = $ARGV[1];
my $out_pre = (defined $ARGV[2]) ? $ARGV[2] : $ARGV[1];

# Todo:
#       1) Support nested <ALT>'s in hypothesis ctm files
#       2) Support multiple reference word paths
#	5) Output IGNORE alignment for non-scoring regions
#	7) Match hyp words to reference by audiofile channels, not speakers
#	6) Det plot output of confidence scores?
#	12) Output #Snt, #S.Err ?  Std. Dev. / Median of speaker statistics?
#	13) Worry about GB, EUC and non-UTF8 unicode encoded stm/ctm files
#		(Use perl's Encode support w/ command line option?)
#	16) Join together close reference utterances for alignment step
#       4) intersegment gaps? => fine as is.
# 	17) Check that start_times are increasing across the hyp lattice
# 	    from left to right.
#	18) Check that <ALT> tags have * as start time and duration --
#	    otherwise they might be words :).

# Some constants.  Many of these should eventually become command line options.

my $fake_region_id = "-9999";
my $insert_cost = 3;
my $delete_cost = 3;
my $sub_cost = 4;
my $opt_del_cost = 2;	# It's better to substitute versus (%HEST)
			# than to insert and then get it correct.

my $log_zero = -1000;		# a fair approximation to -infinity,
				# given that confidence values are typically
				# only precise to at most 3 digits.
my $max_cost = 99999999;

my $allow_multiple_reference_paths = 0;
my $allow_word_fragment_matching = 1;

my $verbosity = 1;
my $debug = 0;

# Globals
my @cost;
my @traceback;

# Load reference
print "Loading $ref_format reference file $ref ...\n" if ( $verbosity > 0 );
my @load_ref_data;
if ($ref_format eq "stm") {
    @load_ref_data = load_stm( $ref );
}
elsif (($ref_format eq "trn") or ($ref_format eq "kaldi")) {
    @load_ref_data = load_snor( $ref, $ref_format );
}
else {
    die "Internal error: invalid format $ref_format";
}
my ( $refreg, $label_names, $category_names, $reforder, $get_side_from_speaker ) = @load_ref_data;

# Merge reference regions into scoring regions
# my $score_regions = merge_ref_regions( $ref_regions );

# Load hypothesis
print "Loading $hyp_format hypothesis file $hyp ...\n" if ( $verbosity > 0 );
my @load_hyp_data;
if ($hyp_format eq "ctm") {
    @load_hyp_data = load_ctm( $hyp, $get_side_from_speaker );
}
elsif (($hyp_format eq "trn") or ($hyp_format eq "kaldi")) {
    my ( $hypreg ) = load_snor( $hyp, $hyp_format );
    @load_hyp_data = ( $hypreg, 5 );   # indicates no confidence field
    # Make word networks for all utterances
    map { MakeNetworkFromText( $_ ) } map { values %$_ } values %$hypreg;
}
else {
    die "Internal error: invalid format $hyp_format";
}
my ( $hypreg, $num_ctm_fields ) = @load_hyp_data;

# Assign hypothesis words to scoring regions (currently done in load_ctm)
# my $assigned_hyps = assign_hyp_words_to_regions( $hyp_words, $score_regions );

# Do the alignment
# my $stats = align( $score_regions, $assigned_hyps, $out_pre );
my $stats = align();

# Print stats
print_stats( $stats, $out_pre );

print STDOUT "Output files written to ${out_pre}.pra (alignments), ${out_pre}.sys (statistics), ${out_pre}.dtl.*\n";

exit(0);

#################################################################################
#				Subroutines
#################################################################################

sub align {

  my $stats = {};
 
  my %sub_count = ();
  my %ins_count = ();
  my %del_count = ();
  my %ref_correct_count = ();
  my %hyp_correct_count = ();
  my %ref_count = ();
  my %hyp_count = ();
  my %ref_sub_count = ();
  my %hyp_sub_count = ();
 
  open ( F, ">" . $out_pre . ".pra" )
  	or die "Couldn't open ${out_pre}.pra for writing alignments\n";
  open ( SF, ">" . $out_pre . ".sgml" )
        or die "Couldn't open ${out_pre}.sgml for writing sgml alignments\n";

  my $date = `date`;
  chomp $date; 
  print SF '<SYSTEM title="' . $out_pre . '" ref_fname="' . $ref . '" hyp_fname="' . $hyp . '" creation_date="' . $date . '" format="2.4" frag_corr="TRUE" opt_del="TRUE" weight_ali="FALSE" weight_filename="">' . "\n";

  foreach my $label ( sort keys %{ $label_names } ) {
    print SF '<LABEL id="' . $label . '" title="' . $label_names->{$label}->{short} . '" desc="' . $label_names->{$label}->{long} . '">' . "\n";
    print SF "</LABEL>\n";
  }

  foreach my $category ( sort { $a <=> $b } keys %{ $category_names } ) {
    print SF '<CATEGORY id="' . $category . '" title="' . $category_names->{$category}->{short} . '" desc="' . $category_names->{$category}->{long} . '">' . "\n";
    print SF "</CATEGORY>\n";
  }

  foreach my $spkr ( @$reforder ) {
      if (not $require_all_in_ref) {
          next unless exists($hypreg->{$spkr});
      }

    print "Aligning $spkr ...\n" if ( $verbosity > 1 );
    print SF '<SPEAKER id="' . $spkr . '">' . "\n";

    my $cnt = 1;
    ALIGN_UTT: foreach my $st ( sort { $a <=> $b } keys %{ $refreg->{$spkr} } ) {
  
      # Align
      my $correct = 0;
      my $insertions = 0;
      my $deletions = 0;
      my $substitutions = 0;
      my $log_prob = 0;
 
      next ALIGN_UTT if ( $refreg->{$spkr}{$st}->{words} =~ /^IGNORE_TIME_SEGMENT_IN_SCORING$/i );
 
      next ALIGN_UTT if ( ( $st eq $fake_region_id ) && !defined($hypreg->{$spkr}{$st}) );

      print "Aligning ${spkr}-${st}\n" if ( $debug );

      # Make the reference lattice from word string
      my $reflat = { arcs => [], nodes => [] };
      push @{ $reflat->{nodes} }, { in_arcs => [], out_arcs => [ 0 ] };
      foreach my $refword ( split( ' ', $refreg->{$spkr}{$st}->{words} ) ) {
        my $last_node_id = $#{ $reflat->{nodes} };
        push @{ $reflat->{arcs} }, { src => $last_node_id,
				     dst => $last_node_id + 1,
				     word => $refword };
        $reflat->{nodes}->[$last_node_id]->{out_arcs} = [ $#{ $reflat->{arcs} } ];
        push @{ $reflat->{nodes} }, { in_arcs => [ $#{$reflat->{arcs} } ], 
				      out_arcs => [] };
      }

      my $hyplat = defined( $hypreg->{$spkr}{$st} ) ?
			$hypreg->{$spkr}{$st} : 
			{ nodes => [ { in_arcs => [], out_arcs => [] } ], 
			  arcs => [] };

      if ( $debug ) {
        print "Reference lattice =\n";
        print_lattice( $reflat );
        print "Hypothesis lattice =\n";
        print_lattice( $hyplat );
      }

      @cost = ();
      @traceback = ();
      $cost[0][0] = 0;
      $traceback[0][0] = {};

      # Assign lowest costs to every ( ref_lat_node, hyp_lat_node ) pair

      for ( my $i = 0; $i <= $#{ $reflat->{nodes} }; $i++ ) {
        HYP_NODES: for ( my $j = 0; $j <= $#{ $hyplat->{nodes} }; $j++ ) {

          next HYP_NODES if ( ( $i == 0 ) && ( $j == 0 ) );

          $cost[$i][$j] = $max_cost;
	  print "Aligning $i,$j\n" if ( $debug );

          foreach my $ref_arc ( @{ $reflat->{nodes}->[$i]->{in_arcs} } ) {
            my $ref_arc_hash = $reflat->{arcs}->[$ref_arc];
            my $ref_word = $ref_arc_hash->{word};

            foreach my $hyp_arc ( @{ $hyplat->{nodes}->[$j]->{in_arcs} } ) {
              my $hyp_arc_hash = $hyplat->{arcs}->[$hyp_arc];

              my $base_cost = $cost[ $ref_arc_hash->{src} ][ $hyp_arc_hash->{src} ];
              my $hyp_word = $hyp_arc_hash->{word};
              my $move_cost;
              my $tb_str;

              print "Comparing ref $ref_word vs. hyp $hyp_word\n" if ( $debug );

              if ( $ref_word eq $hyp_word ) {
                 $move_cost = $base_cost;
                 $tb_str = "CORRECT: $ref_word";
              } elsif ( ( $ref_word eq "(" . $hyp_word . ")" ) ||
                   ( $allow_word_fragment_matching &&
                     ( ( ( $ref_word =~ /^\((.*)\-\)$/ ) &&     # (X-) can match XY
                         ( $hyp_word =~ /^$1/ ) ) ||
                       ( ( $ref_word =~ /^\(\-(.*)\)$/ ) &&     # (-X) can match YX
                         ( $hyp_word =~ /$1$/ ) ) ) ) ){
                 $move_cost = $base_cost;
                 $tb_str = "CORRECT: hyp $hyp_word for ref $ref_word";
              } else {
                 $move_cost = $base_cost + $sub_cost;
                 $tb_str = "SUBSTITUTION: hyp $hyp_word for ref $ref_word";
              }

              update_cost( $i, $j, $ref_arc, $hyp_arc, $move_cost, $tb_str );
            }

            # Deletions
            my $base_cost = $cost[ $ref_arc_hash->{src} ][ $j ];
	    my $move_cost;
            my $tb_str;
            if ( $ref_word =~ /^\(.*\)$/ ) {
              $move_cost = $base_cost + $opt_del_cost;
              $tb_str = "CORRECT (Opt. Del.): $ref_word";
            } else {
              $move_cost = $base_cost + $delete_cost;
              $tb_str = "DELETION: $ref_word";
            }

            update_cost( $i, $j, $ref_arc, undef, $move_cost, $tb_str );

          }

          # Insertions
          foreach my $hyp_arc ( @{ $hyplat->{nodes}->[$j]->{in_arcs} } ) {
            my $hyp_arc_hash = $hyplat->{arcs}->[$hyp_arc];
            my $base_cost = $cost[$i][ $hyp_arc_hash->{src} ];
            my $hyp_word = $hyp_arc_hash->{word};

            my $move_cost = $base_cost + $insert_cost;
            my $tb_str = "INSERTION: $hyp_word";
            update_cost( $i, $j, undef, $hyp_arc, $move_cost, $tb_str );
          }

        } # for $j
      } # for $i
  
      # Traceback
      my $i = $#{ $reflat->{nodes} };
      my $j = $#{ $hyplat->{nodes} };
      my $aligned_ref = "";
      my $aligned_hyp = "";
      my $align_str = "";
      my $sgml_str = "";
      
      while ( ( $i > 0 ) || ( $j > 0 ) ) {
        my $tb = $traceback[$i][$j];

#        print "Traceback for $i,$j is $tb" if ( $debug );

        my $tb_str = $tb->{str};
        die "Undefined traceback string for speaker $spkr start time $st (i=$i,j=$j)" unless defined( $tb_str );

        $align_str = $tb_str . "\n" . $align_str;

        my $ref_arc_hash = defined($tb->{ref_arc}) ? $reflat->{arcs}->[$tb->{ref_arc}] : {};
        my $hyp_arc_hash = defined($tb->{hyp_arc}) ? $hyplat->{arcs}->[$tb->{hyp_arc}] : {};
	my $ref_word = defined( $ref_arc_hash->{word} ) ? $ref_arc_hash->{word} : "";
        my $hyp_word = defined( $hyp_arc_hash->{word} ) ? $hyp_arc_hash->{word} : "";
	my $hyp_word_conf = defined( $hyp_arc_hash->{conf} ) ? $hyp_arc_hash->{conf} : "";
        my $hyp_start_time = defined( $hyp_arc_hash->{start_time} ) ? $hyp_arc_hash->{start_time} : "";
        my $hyp_end_time = defined( $hyp_arc_hash->{end_time} ) ? $hyp_arc_hash->{end_time} : "";

        if ( $ref_word ) {
          $aligned_ref = $ref_word . " " . $aligned_ref;
          $ref_count{$ref_word} = 0 unless defined( $ref_count{$ref_word} );
          $ref_count{$ref_word} += 1;
        }

        if ( $hyp_word ) {
          $aligned_hyp = $hyp_word . " " . $aligned_hyp;
          $hyp_count{$hyp_word} = 0 unless defined( $ref_count{$hyp_word} );
          $hyp_count{$hyp_word} += 1;
        }

	my $next_i = defined($ref_arc_hash->{src}) ? $ref_arc_hash->{src} : $i;
        my $next_j = defined($hyp_arc_hash->{src}) ? $hyp_arc_hash->{src} : $j;

        if ( $tb_str =~ /^C/ ) {
          $correct += 1;
          $ref_correct_count{$ref_word} = 0 
		unless defined( $ref_correct_count{$ref_word} );
          $ref_correct_count{$ref_word} += 1;
          if ( $tb_str !~ /^CORRECT \(Opt/ ) {
            $log_prob += mylog( $hyp_word_conf );
            $sgml_str = 'C,"' . $ref_word . '","' . $hyp_word . '",' . $hyp_start_time . '+' . $hyp_end_time . ',' . $hyp_word_conf . ':' . $sgml_str;
            $hyp_correct_count{$hyp_word} = 0
                unless defined( $hyp_correct_count{$hyp_word} );
	    $hyp_correct_count{$hyp_word} += 1;
          } else {
            $sgml_str = 'C,"' . $ref_word . '","",0.000+0.000,0.000000:' . $sgml_str;
          }
        } elsif ( $tb_str =~ /^S/ ) {
          $substitutions +=1;
          $log_prob += mylog( 1.0 - $hyp_word_conf );
          $sgml_str = 'S,"' . $ref_word . '","' . $hyp_word . '",' 
			. $hyp_start_time . '+' . $hyp_end_time . ',' 
			. $hyp_word_conf . ':' . $sgml_str;
          $sub_count{"$ref_word $hyp_word"} = 0
		unless defined( $sub_count{"$ref_word $hyp_word"} );
          $sub_count{"$ref_word $hyp_word"} += 1;
          $ref_sub_count{$ref_word} = 0
                unless defined( $ref_sub_count{$ref_word} );
          $ref_sub_count{$ref_word} += 1;
          $hyp_sub_count{$hyp_word} = 0
                unless defined( $hyp_sub_count{$hyp_word} );
          $hyp_sub_count{$hyp_word} += 1;
        } elsif ( $tb_str =~ /^I/ ) {
          $insertions += 1;
          $log_prob += mylog( 1.0 - $hyp_word_conf );
          $sgml_str = 'I,,"' . $hyp_word . '",' . $hyp_start_time . '+' . 
			$hyp_end_time . ',' . $hyp_word_conf 
			. ':' . $sgml_str;
          $ins_count{$hyp_word} = 0 unless defined( $ins_count{$hyp_word} );
          $ins_count{$hyp_word} += 1;
        } elsif ( $tb_str =~ /^D/ ) {
          $deletions += 1;
          $sgml_str = 'D,"' . $ref_word . '",,,:' . $sgml_str;
          $del_count{$ref_word} = 0 unless defined( $del_count{$ref_word} );
          $del_count{$ref_word} += 1;
        } else {
          die "INTERNAL ERROR:  Unknown traceback string $tb_str while aligning speaker $spkr reference starting at $st\n";
        }

      $i = $next_i;
      $j = $next_j;
      } # end while
  
      my $et = $refreg->{$spkr}{$st}->{end_time};
      if ( $st eq $fake_region_id ) {
        print F "Speaker $spkr  Hypothesis words outside of reference regions\n";
        }
      else {
          if (($ref_format eq 'trn') or ($ref_format eq 'kaldi')) {
              print F "id: ${spkr}${SPEAKER_SIDE_DELIM}$st\n";
          }
          else {
              print F "Speaker $spkr Start time $st  End time $et\n";
          }
        }
      print F "Ref: $aligned_ref\n";
      print F "Hyp: $aligned_hyp\n";

      print F "Scores: ( #C #S #D #I ) = ( $correct $substitutions $deletions $insertions )\n";
      print F $align_str;
      print F "\n";

      my $nreference = $correct + $substitutions + $deletions;
      my $nhypothesis = $correct + $substitutions + $insertions;

      print SF '<PATH id="(' . $spkr . "-" . $st . "-" . $et . 
	       ')" word_cnt="' . $nhypothesis . 
	 	'" labels="<' .  $refreg->{$spkr}{$st}->{tags} . 
		'>" file="' .  $refreg->{$spkr}{$st}->{wavefile} .
	       '" channel="' . $refreg->{$spkr}{$st}->{channel} .  
		'" sequence="' . $cnt++ . 
               '" R_T1="' . $st . '" R_T2="' . $et . 
               '" word_aux="h_t1+t2,h_conf">' . "\n";
      chop $sgml_str;
      print SF $sgml_str . "\n";
      print SF "</PATH>\n";

      # Accumulate statistics
      foreach my $t ( split( ',', $refreg->{$spkr}{$st}->{tags} ), 
  		    $refreg->{$spkr}{$st}->{speaker}, 
                      "ALL" ) {
        my $s = { nref => $nreference,
  		  nhyp => $nhypothesis,
                  cor => $correct,
                  sub => $substitutions,
                  ins => $insertions,
                  del => $deletions,
                  logprob => $log_prob };
        foreach my $k ( keys %{ $s } ) {
          $stats->{$t}->{$k} += $s->{$k};
        }
      }
  
    } # $foreach $st

    print SF "</SPEAKER>\n";
  } # foreach $spkr

  close( F );

  print SF "</SYSTEM>\n";
  close( SF );

  # dtl files

  open ( DTL, ">" . $out_pre . ".dtl.sub" )
        or die "Couldn't open ${out_pre}.dtl.sub for writing substitution counts\n";
  print DTL "Substitutions\n\n";
  print DTL "Count  Ref_word  Hyp_word\n";
  print DTL "---------------------------------------\n";
  foreach my $k ( sort { $sub_count{$b} <=> $sub_count{$a} } keys %sub_count ) {
     printf DTL "%5d  %-70s\n", $sub_count{$k}, $k;
  } 
  close( DTL );

  open ( DTL, ">" . $out_pre . ".dtl.ins" )
        or die "Couldn't open ${out_pre}.dtl.ins for writing insertion counts\n";
  print DTL "Insertions\n\n";
  print DTL "Count  Hyp_word\n";
  print DTL "---------------------------------------\n";
  foreach my $k ( sort { $ins_count{$b} <=> $ins_count{$a} } keys %ins_count ) {
     printf DTL "%5d  %-70s\n", $ins_count{$k}, $k;
  }
  close( DTL );

  open ( DTL, ">" . $out_pre . ".dtl.del" )
        or die "Couldn't open ${out_pre}.dtl.del for writing deletion counts\n";
  print DTL "Deletions\n\n";
  print DTL "Count  Ref_word\n";
  print DTL "---------------------------------------\n";
  foreach my $k ( sort { $del_count{$b} <=> $del_count{$a} } keys %del_count ) {
     printf DTL "%5d  %-70s\n", $del_count{$k}, $k;
  }
  close( DTL );

  open ( DTL, ">" . $out_pre . ".dtl.ref_words" )
        or die "Couldn't open ${out_pre}.dtl.ref_words for writing reference word statistics\n";
  print DTL "Statistics by reference word\n\n";
  printf DTL "%-25s  %6s  %4s  %4s  %4s  %4s\n",
	"Word", "Count", "%Cor", "%Err", "%Sub", "%Del";
  print DTL "---------------------------------------------------------\n";
  foreach my $k ( sort { $ref_count{$b} <=> $ref_count{$a} } keys %ref_count ) {
     $ref_correct_count{$k} = 0 unless defined( $ref_correct_count{$k} );
     $ref_sub_count{$k} = 0 unless defined( $ref_sub_count{$k} );
     $del_count{$k} = 0 unless defined( $del_count{$k} );
     printf DTL "%-25s  %6s  %4d  %4d  %4d  %4d\n",
		$k, $ref_count{$k}, 
	        100 * ( $ref_correct_count{$k} / $ref_count{$k} ),
		100 * ( $ref_sub_count{$k} + $del_count{$k} ) / $ref_count{$k},
		100 * ( $ref_sub_count{$k} / $ref_count{$k} ), 
	        100 * ( $del_count{$k} / $ref_count{$k} );
  }
  close( DTL );

  open ( DTL, ">" . $out_pre . ".dtl.hyp_words" )
        or die "Couldn't open ${out_pre}.dtl.hyp_words for writing reference word statistics\n";
  print DTL "Statistics by hypothesis word\n\n";
  printf DTL "%-25s  %6s  %4s  %4s  %4s  %4s\n",
        "Word", "Count", "%Cor", "%Err", "%Sub", "%Ins";
  print DTL "---------------------------------------------------------\n";
  foreach my $k ( sort { $hyp_count{$b} <=> $hyp_count{$a} } keys %hyp_count ) {
     $hyp_correct_count{$k} = 0 unless defined( $hyp_correct_count{$k} );
     $hyp_sub_count{$k} = 0 unless defined( $hyp_sub_count{$k} );
     $ins_count{$k} = 0 unless defined( $ins_count{$k} );
     printf DTL "%-25s  %6s  %4d  %4d  %4d  %4d\n",
                $k, $hyp_count{$k},
                100 * ( $hyp_correct_count{$k} / $hyp_count{$k} ),
                100 * ( $hyp_sub_count{$k} + $ins_count{$k} ) / $hyp_count{$k},
                100 * ( $hyp_sub_count{$k} / $hyp_count{$k} ),
                100 * ( $ins_count{$k} / $hyp_count{$k} );
  }
  close( DTL );


  return $stats;

} # end of align


sub print_stats {

  my ( $stats, $out_pre ) = @_;

  my $sysf = $out_pre . ".sys";
  my $rawf = $out_pre . ".raw";
  
  if ( $verbosity > 1 ) {
    open( F, "| tee $sysf" ) or
  	die "Couldn't open | tee $sysf for writing\n";
    }
  else {
    open( F, ">" . $sysf ) or
  	die "Couldn't open $sysf for writing\n";
    }
  
  open( RAW, ">" . $rawf ) or
  	die "Couldn't open $rawf for writing\n";
  
  my $format = "%15s  %6s  %6s  %6s  %5s  %5s  %5s  %5s  %7s\n";
  my $dash_line = ("-" x 79) . "\n";
  
  printf F $format, "Label", "#Ref", "#Hyp", "WER", "%Cor", "%Sub", "%Del", "%Ins", "NCE";
  print F $dash_line;
 
  printf RAW $format, "Label", "#Ref", "#Hyp", "#Err", "#Cor", "#Sub", "#Del", "#Ins", "NCE";
  print RAW $dash_line;
 
  foreach my $t ( sort keys %{ $stats } ) {
    my $label = $t;
    $label = $label_names->{$t}->{short} if defined( $label_names->{$t}->{short} );
  
    my $st = $stats->{$t};
  
    # Prevent divide by zero
    $st->{nref} = 1 if ($st->{nref} == 0);

    my $wer = ( $st->{sub} + $st->{del} + $st->{ins} ) / $st->{nref} * 100.0;
    my $p_c = ( $st->{nhyp} == 0 ) ? 0.5 : $st->{cor} / $st->{nhyp};
    my $h_max = -$st->{cor} * mylog($p_c) - ($st->{nhyp} - $st->{cor}) * mylog( 1 - $p_c );
    my $NCE = ( $h_max == 0.00 ) ? "XXX" : sprintf( "%7.3f", 1.0 + $st->{logprob} / $h_max ); 
    $NCE = "n/a" if ( $num_ctm_fields == 5 );
  
    print F $dash_line if ( $t eq "ALL" );
    print RAW $dash_line if ( $t eq "ALL" );
  
    printf F $format, $label, $st->{nref}, $st->{nhyp}, sprintf( "%6.2f", $wer ),
  	sprintf( "%5.1f", $st->{cor} / $st->{nref} * 100.0 ), 
          sprintf( "%5.1f", $st->{sub} / $st->{nref} * 100.0 ),
          sprintf( "%5.1f", $st->{del} / $st->{nref} * 100.0 ),
          sprintf( "%5.1f", $st->{ins} / $st->{nref} * 100.0 ), 
  	$NCE;
  
    printf RAW $format, $label, $st->{nref}, $st->{nhyp}, 
  	( $st->{sub} + $st->{del} + $st->{ins} ),
          $st->{cor}, $st->{sub}, $st->{del}, $st->{ins}, $NCE;
  
    print F $dash_line if ( $t eq "ALL" );
    print RAW $dash_line if ( $t eq "ALL" );
  
    }
  print F $dash_line;
  printf F $format, "Label", "#Ref", "#Hyp", "WER", "%Cor", "%Sub", "%Del", "%Ins", "NCE";
 
  print RAW $dash_line; 
  printf RAW $format, "TAG", "#Ref", "#Hyp", "#Err", "#Cor", "#Sub", "#Del", "#Ins", "NCE";
  
  close( F );
  close( RAW );

  return;
}  
  
sub mylog
{
  my $x = shift;

  return ( $x > 0 ) ? log( $x) : $log_zero;
}
  
sub load_stm
{
  my $ref = shift;

  my $refreg = {};
  my $label_names = {};
  my $category_names = {};
  my $reforder = [];
  my $side_from_speaker = 0;

  open( R, $ref ) or die "Can't open stm file $ref for reading\n";
  REF: while( <R> ) {

    if ( /^;;\s*LABEL\s*\"([^\"]*)\"\s*\"([^\"]*)\"\s*\"([^\"]*)\"/ ) {
      my $label = $1;
      warn "Previously defined label $label is redefined on STM file $ref line $_\n"
  	if ( defined( $label_names->{$label} ) 
	     && ( $label_names->{$label}->{short} ne $2 ) );
      die "Label '$label' may not contain spaces on STM file $ref line $_\n"
  	if ( $label =~ / / );
      $label_names->{$label} = { short => $2, long => $3 };
      }

    if ( /^;;\s*CATEGORY\s*\"([^\"]*)\"\s*\"([^\"]*)\"\s*\"([^\"]*)\"/ ) {
      my $category = $1;
      warn "Previously defined category $category is redefined on STM file $ref line $_\n"
        if ( defined( $category_names->{$category} )
             && ( $category_names->{$category}->{short} ne $2 ) );
      die "Category '$category' may not contain spaces on STM file $ref line $_\n" if ( $category =~ / / );
      $category_names->{$category} = { short => $2, long => $3 };
      }

    next REF if (/^;/ or /^\s*$/);
  
    my @f = split;
    (@f >= 6) or die "Stm file $ref line\n$_ doesn't have enough fields. (It must have wavefile channel speaker start_time end_time tag)\n";
    my ($wavefile) = fileparse($f[0], qr/\.[^.]*$/);
    my $channel = $f[1];
    my $speaker = $f[2];
    my $start_time = $f[3];
    my $end_time = $f[4];
    my $tag = $f[5];
    $tag =~ /^\<(.*)\>$/ or die "Couldn't parse tag field $tag of stm file $ref line $_ ; tag field must start with < and end with >\n";
    $tag = $1;
    my $words = join( ' ', @f[6 .. $#f] );
  
    if ( $end_time < $start_time + 0.0001 ) {
      print "WARNING:  For stm file $ref line $_ the end time $end_time isn't after the start time $start_time plus 0.0001 .\n";
      $end_time = $start_time + 0.0001;
      }
   
    # For first cut, stm file utterances = scorable regions
    my $side;
    if ($channel =~ /^[A-Z]$/) {  # usually will be just 'A' or 'B', but some intermediate scripts can have a channel of 'X'
        $side = $wavefile . "_" . $channel;
    }
    else {
        $side = $channel;
        $side_from_speaker = 1;
    }

    push(@$reforder, $side) unless $refreg->{$side};

    # Check for overlapping reference regions
    foreach my $st ( keys %{ $refreg->{$side} } ) {
       my $et = $refreg->{$side}{$st}->{end_time};
       if ( ( ( $start_time > $st ) && ( $start_time < $et ) )
            || ( $end_time > $st ) && ( $end_time < $et ) ) {
         warn "STM line $_ overlaps with STM utterance starting at $st and ending at $et\n\n";
       }
    }

    $refreg->{$side}{$start_time} = { end_time => $end_time,
				      tags => $tag,
				      words => $words,
				      speaker => $speaker,
				      wavefile => $wavefile,
				      channel => $channel };
    $refreg->{$side}{$fake_region_id} = { end_time => $fake_region_id,
					  tags => $tag,
					  words => "",
					  speaker => $speaker,
                                          wavefile => $wavefile,
                                          channel => $channel }
  	unless defined( $refreg->{$side}{$fake_region_id} );
  
    }
  close( R );

  return ( $refreg, $label_names, $category_names, $reforder, $side_from_speaker );
}

sub load_snor
{
    my $filename = shift;
    my $fmt      = shift;

    my $reg = {};
    my $label_names = {};
    my $category_names = {};
    my $order = [];

    open( R, $filename ) or die "Can't open trn file $filename for reading\n";
    RECORD: while( <R> ) {
        chomp;
        next RECORD if (/^\s*$/);
        
        my @f = split;
        next RECORD unless @f;

        my $snorIdField;
	if ($fmt eq "trn") {
	    $snorIdField = pop(@f);
	    $snorIdField = StripParens($snorIdField);
	} elsif ($fmt eq "kaldi") {
	    $snorIdField = shift(@f);
	} else {
	    die "load_snor(): unknown format \"$fmt\"! ";
	}
        my $side;
        my $uttIndex;
        
        if ($snorIdField =~ m/^(\S+)([_-])(\d+)$/) {
            $side = $1;
	    $SPEAKER_SIDE_DELIM = $2;
            $uttIndex = $3;
        } else {
            $side = $snorIdField;
            $uttIndex = "1";
        }

        unless (defined($side) and defined($uttIndex)) {
            die "Transcript (SNOR) file $filename bad SNOR id $snorIdField on line $_\n";
        }

        my $words = join( ' ', @f);
  
        push(@$order, $side) unless $reg->{$side};

        # Check for repeated index within speaker
        if ( $reg->{$side}{$uttIndex} ) {
            warn "Transcript (SNOR) line side $side index $uttIndex repeated\n\n";
       }
# XXXX later need to decide exactly what field values will be
        $reg->{$side}{$uttIndex} = {   end_time => "XXX",
                                       tags => "",
                                       words => $words,
                                       speaker => $side,
                                       wavefile => "XXX",
                                       channel => "XXX" };
# XXXX do we need fake_region_id?
# XXXX MD disabling
        # $reg->{$side}{$fake_region_id} = { end_time => $fake_region_id,
        #                                    tags => "",
        #                                    words => "",
        #                                    speaker => $side,
        #                                    wavefile => "XXX",
        #                                    channel => "XXX", }
        # unless defined( $reg->{$side}{$fake_region_id} );
        
    }
    close( R );

# Note: label_names, category_names always empty for SNOR file
  return ( $reg, $label_names, $category_names, $order, 0 );
}

sub StripParens {
    my ($str) = @_;
    # Return contents of one-level of matched, enclosing parentheses
    # Strips the parens and adjoining space
    # (More general than needs to be: allows internal blanks
    # in the contained string, e.g.,
    #    " (   Hi there )" -->  "Hi there"
    # if ( $str =~ /^\s*\(\s*(\S+(\s+\S+)*)\s*\)\s*$/ ) {
    # Changed my mind, just keep it simple, allow no spaces:
    if ( $str =~ /^\((\S+)\)$/ ) {
        return $1;
    }
    return undef;
}

sub ParseSNORID {
    my ($snorId) = @_;
    # This must stupidly assume that ids are in form side-uttindex, eg, sw2001-A-0001.
    if ( $snorId && $snorId =~ /^(\S+)-(\d+)$/ ) {
        return ($1, $2);
    }
    return (undef, undef);
}

sub MakeNetworkFromText {
    my ($lat) = @_;

    @$lat{"arcs", "nodes"} = ( [], [] );
    push @{ $lat->{nodes} }, { in_arcs => [], out_arcs => [ 0 ] };
    my $words = $lat->{words};
    return unless $words;

    foreach my $word ( split( ' ', $words) ) {
        my $last_node_id = $#{ $lat->{nodes} };
        push @{ $lat->{arcs} }, { src => $last_node_id,
                                  dst => $last_node_id + 1,
                                  word => $word,
                                  conf => 0.5 };
        $lat->{nodes}->[$last_node_id]->{out_arcs} = [ $#{ $lat->{arcs} } ];
        push @{ $lat->{nodes} }, { in_arcs => [ $#{$lat->{arcs} } ], 
				      out_arcs => [] };
    }
}

sub check_conf
{
  my ( $hyp, $line, $conf, $num_ctm_fields ) = @_;

  if ( defined( $conf ) ) {
    # check that conf value is valid, if it isn't verify that the decoding type is being called correctly, especially DecodeFastFWBW
    die "On ctm file $hyp line $line confidence value $conf isn't valid numeric value." unless ( $conf =~ /^\s*[-+]?[0-9]*(?:[0-9]|\.[0-9]*)?(?:[eE][-+]?[0-9]+)?\s*$/);

    if ( $num_ctm_fields == 5 ) {
        warn "CTM file $hyp started out having five fields, but line $line has six!\n";
      } else {
        $num_ctm_fields = 6;
      }
    } else {
      $conf = 0.5;
      if ( $num_ctm_fields == 6 ) {
        warn "CTM file $hyp started out having six fields, but line $line has only five!\n";
      } else {
        $num_ctm_fields = 5;
      }
    }

    die "On ctm file $hyp line $line confidence value $conf isn't between 0 and 1\n"
        if ( ( $conf > 1.0 ) || ( $conf < 0.0 ) );

  return ( $conf, $num_ctm_fields );
}

sub load_ctm 
{
  my $hyp = shift;
  my $side_from_speaker = shift;

  my $hypreg = {}; 
  my $num_ctm_fields = 0;
  my $curr_spkr = undef;

  open( H, $hyp ) or die "Can't open ctm file $hyp for reading\n";
  HYP: while( <H> ) {
    #next HYP if ( /^[;#]/ or /^\s*$/ );
    if ( /^[;#]/ ) {
        # Save the speaker ID from the comment that starts a new utterance.  If the STM was
        # indexed by speaker ID, we will use this to look up the matching reference transcription.
        if ( /spkr (\S+)/ ) {
            $curr_spkr = $1;
        }
        next HYP;
    }
    next HYP if ( /^\s*$/ );

    my ($wavefile, $channel, $start_time, $duration, $word, $conf, @foo) = split;
    (defined($word) && !(@foo)) or die "Ctm file $hyp line $_ doesn't have five or six fields\n(It must have wavefile channel start_time end_time word [confidence].)\n";

    # Extract id from file
    $wavefile = fileparse($wavefile, qr/\.[^.]*$/);

    # Assign it a scorable region
    my $side;
    if ($side_from_speaker && defined($curr_spkr)) {
        $side = $curr_spkr;
    }
    else {
        $side = $wavefile . "_" . $channel;
    }
 
    if ( $word eq "<ALT_BEGIN>" ) {
      my $orig_wavefile = $wavefile;
      my $orig_channel = $channel;
      my @alt_hyps = ( [] );
      my $i = 0;

      my $region_start = 99999999; 
      my $region_end = -99999;

      ALT_LINE: while ( <H> ) {
        my ($wavefile2, $channel2, $start_time2, $duration2, $word2, $conf2, @foo2) = split;
        $wavefile2 = fileparse($wavefile2, qr/\.[^.]*$/);
        die "Wavefile switched from $orig_wavefile to $wavefile2 inside <ALT> block at ctm file $hyp line $_" unless ( $orig_wavefile eq $wavefile2);
        die "Channel switched from $orig_channel to $channel2 inside <ALT> block at ctm file $hyp line $_" unless ( $orig_channel eq $channel2 );

        (defined($word2) && !(@foo2)) or die "Ctm file $hyp line $_ doesn't have five or six fields\n(It must have wavefile channel start_time end_time word [confidence].)\n";

        if ( $word2 eq "<ALT>" ) {
          $i++; 
          $alt_hyps[$i] = [];
          next ALT_LINE;
        }
        if ( $word2 eq "<ALT_END>" ) {
          last ALT_LINE;
        }

        ($conf2,$num_ctm_fields) = check_conf( $hyp, $_, $conf2, $num_ctm_fields);

        push @{ $alt_hyps[$i] }, [$start_time2, $start_time2 + $duration2, $word2, $conf2];

        $region_start = min( $region_start, $start_time2 );
        $region_end = max( $start_time2 + $duration2, $region_end );
      }

      # Put the <ALT> block into a hypreg
      my $best_region = find_best_region( $side, $region_start, $region_end );

      $hypreg->{$side}{$best_region} = { arcs => [],
                                       nodes => [ { in_arcs => [],
                                                  out_arcs => [] } ] }
        unless defined( $hypreg->{$side}{$best_region} );

      my $alt_start_node_id = $#{ $hypreg->{$side}{$best_region}->{nodes} };
      my @arc_ids_to_fix = ();

      foreach my $i ( 0 .. $#alt_hyps ) {
         my $start_node_id = $alt_start_node_id;
         foreach my $j ( 0 .. $#{ $alt_hyps[$i] } ) {

            push @{ $hypreg->{$side}{$best_region}->{arcs} },
        	{ word => $alt_hyps[$i]->[$j]->[2], 
		  conf => $alt_hyps[$i]->[$j]->[3], 
		  start_time => $alt_hyps[$i]->[$j]->[0],
		  end_time => $alt_hyps[$i]->[$j]->[1],
		  src => $start_node_id,
		  dst => $#{ $hypreg->{$side}{$best_region}->{nodes} } + 1,
		};
           
            push @{ $hypreg->{$side}{$best_region}->{nodes}->[$start_node_id]->{out_arcs} }, $#{ $hypreg->{$side}{$best_region}->{arcs} };

            if ( $j != $#{ $alt_hyps[$i] } ) {
               push @{ $hypreg->{$side}{$best_region}->{nodes} },
		{ in_arcs => [ $#{ $hypreg->{$side}{$best_region}->{arcs} } ], 
	          out_arcs => [] };
               $start_node_id = $#{ $hypreg->{$side}{$best_region}->{nodes} };
            } else {
               push @arc_ids_to_fix, 
		    $#{ $hypreg->{$side}{$best_region}->{arcs} };
            }

         } # foreach $j
      } # foreach $i

    push  @{ $hypreg->{$side}{$best_region}->{nodes} },
        { in_arcs => [ @arc_ids_to_fix ], out_arcs => [] };
    foreach my $arc_id ( @arc_ids_to_fix ) {
       $hypreg->{$side}{$best_region}->{arcs}->[$arc_id]->{dst} =
	$#{ $hypreg->{$side}{$best_region}->{nodes} };
    }

    next HYP;
    } # if $word eq "<ALT_BEGIN>"
 
    my $end_time = $start_time + $duration;

    ($conf,$num_ctm_fields) = check_conf( $hyp, $_, $conf, $num_ctm_fields);

    my $best_region = find_best_region( $side, $start_time, $end_time );
 
    $hypreg->{$side}{$best_region} = { arcs => [], 
				       nodes => [ { in_arcs => [], 
						  out_arcs => [] } ] }
	unless defined( $hypreg->{$side}{$best_region} );

    my $last_node_id = $#{ $hypreg->{$side}{$best_region}->{nodes} };
    push @{ $hypreg->{$side}{$best_region}->{arcs} },
	{ src => $last_node_id, dst => $last_node_id + 1, word => $word, 
	  conf => $conf, start_time => $start_time, end_time => $end_time };
    $hypreg->{$side}{$best_region}->{nodes}->[$last_node_id]->{out_arcs}
	= [ $#{ $hypreg->{$side}{$best_region}->{arcs} } ];
    push @{ $hypreg->{$side}{$best_region}->{nodes} },
 	{ in_arcs => [ $#{ $hypreg->{$side}{$best_region}->{arcs} } ], 
	  out_arcs => [] };

    } # while <H>
  close( H );
 
  return( $hypreg, $num_ctm_fields ); 
}
  

sub min {
  my ( $a, $b ) = @_;
  return ( $a < $b ) ? $a : $b;
}

sub max {
  my ( $a, $b ) = @_;
  return ( $a > $b ) ? $a : $b;
}

sub find_best_region {
  my ( $side, $start_time, $end_time ) = @_;

  if ( !defined( $refreg->{$side} ) ) {
    die "Wavefile+channel $side from ctm file $hyp line $_ wasn't seen in the stm file reference.\n";
  }

  my $best_region = $fake_region_id;
  my $dist_to_best_region = 9999999999;

  ST: foreach my $st ( keys %{ $refreg->{$side} } ) {
      next ST if ( $st eq $fake_region_id );
      my $et = $refreg->{$side}{$st}->{end_time};
      my $dist = 0;
      if ( $start_time < $st ) {
        $dist = $st - $start_time;
      }
      if ( $end_time > $et ) {
        $dist += $end_time - $et;
      }
      if ( $dist < $dist_to_best_region ) {
        $best_region = $st;
        $dist_to_best_region = $dist;
        }
      }

  return $best_region;
}

sub update_cost {
  my ( $i, $j, $ref_arc, $hyp_arc, $move_cost, $str ) = @_;

  die "Disconnected lattice at $i, $j $str" if ( $move_cost > $max_cost );

  if ( $move_cost < $cost[$i][$j] ) {
     $cost[$i][$j] = $move_cost;
     $traceback[$i][$j] = { ref_arc => $ref_arc, 
			    hyp_arc => $hyp_arc, 
                            str => $str };
     print "New lowest cost $move_cost for $i,$j with $str\n" if ( $debug );
  }

}

sub print_lattice {
  my ( $lat ) = @_;

  return unless defined( $lat->{nodes} );

  print "Nodes:\n";
  for ( my $n = 0 ; $n <= $#{ $lat->{nodes} }; $n++ ) {
     print "  $n in_arcs = ", join( ' ', @{ $lat->{nodes}->[$n]->{in_arcs} } ),
	   " out_arcs = ", join( ' ', @{ $lat->{nodes}->[$n]->{out_arcs} } ),
	   "\n";
  }
  print "Arcs:\n";
  for ( my $a = 0 ; $a <= $#{ $lat->{arcs} }; $a++ ) {
     print " $a word = ", $lat->{arcs}->[$a]->{word}, " src = ",
	$lat->{arcs}->[$a]->{src}, " dst = ", $lat->{arcs}->[$a]->{dst}, "\n";
  }

}
__END__

=head1 NAME

scorer.pl - Score speech recognition system output

=head1 SYNOPSIS

scorer.pl STM-reference-file CTM-hypothesis-file [output-filename-prefix]

=head1 DESCRIPTION

scorer.pl aligns the words in the CTM-hypothesis-file against the STM-reference-file
and then prints out various statistics of the alignment, including the
word error rate (WER), to standard out and to files beginning with
output-filename-prefix (or CTM-hypothesis-file if output-filename-prefix
is not given).  It is intended as a replacement for sclite(1).

=head1 OPTIONS

Currently, scorer.pl takes no options.

=head1 ALIGNMENT

The alignment process consists of two steps.  In the first step, each word from
the CTM hypothesis file is assigned to an utterance from the STM-reference-file.
In the second step, the reference words in each utterance are aligned with
the hypothesis words assigned to that utterance so as to minimize a
Levenshtein edit-distance function with correct words, insertions, deletions
and substitutions given costs of 0, 3, 3, and 4 respectively.  (Inserting an
optionally deletable word counts as correct, but is given a cost of 2 for
alignment purposes.)

=head1 STM FILE FORMAT

STM (Segment Time Mark) files are text files, any line of which can be either
a blank line, a comment line, a label declaration line, or a regular line.  Blank
and comment lines are ignored.

Comment lines begin with a semicolon character and may then consist of any
number of non-new line characters.  [Note:  sclite requires STM comment lines
to begin with two semicolon characters.]

Label declaration lines begin with two semicolons followed by optional whitespace 
followed by the word "LABEL".  Next comes three strings, each of which is delimmited 
on both ends by double quotes ("), with optional whitespace between the strings.
The first string is the label tag used to mark utterances.  It may not contain spaces.
The second string is the short label description, and is used when presenting
summary statistics for the utterances belonging to the label.  The third string
is a long label description, and is currently unused.

Here are some example label declarations:

   ;; LABEL "F"	   "Female"        "Female Speakers"
   ;; LABEL "FISH" "Fisher"        "Fisher Speakers"
   ;; LABEL "CH-M" "Callhome Male" "Male Callhome Speakers"

Label declarations may be grouped in category sections, which are declared with
lines that look like
   
   ;; CATEGORY "0" "" ""

scorer.pl currently does not use category information in any way.

Regular STM file lines give the transcription and time information for reference
utterances, and consist of at least six whitespace separated fields.  The meaning
of the fields is as follows:

=over 4

=item Field 1:  

Audio file identifier.  Typically this is the basename of the audio
file, without any path information or file type suffixes (like ".sph" or ".wav").

=item Field 2:  

Channel identifier.  Typically "A" for channel 1, and "B" for
channel 2.

=item Field 3:  

Speaker identifier.  Typically this is the audio file identifier
followed by an underscore (_) followed by the channel identifier.

=item Field 4:  

Utterance begin time in seconds, as counted from the beginning of
the audio file.  Typically specified to 1/100ths of a second.

=item Field 5:  

Utterance end time in seconds.

=item Field 6:  

Label tags for this utterance.  The label tags should be separated by commas (,) and enclosed by < and >.  For example:  <F,FISH,FISH-F>.  If there are no label tags
for the utterance, the string <> is expected.  [Note:  unlike in sclite, this field is 
mandatory.]

=item Fields 7+ (Optional):  

The words for this utterance.  Any words enclosed with 
parenthesis are considered to be "optionally deletable":  if no hypothesis word
aligns to the optionally deletable word, then it is counted as correct.  For example,
if "(%HESITATION)" is the sole reference word for an utterance, then either
%HESITATION or no hypothesis for the utterance will assigned 1 correct word and
0 errors for the utterance.  If a single non-%HESITATION word is hypothesized,
it will be counted as a substitution.  If a optionally deletable word ends with 
a dash, then it is considered to be a word fragment and any hypothesis word that 
matches the word upto dash will be considered correct.  For example, the hypothesis
MOLD would be correct if aligned to (MOL-), (MO-), or (M-).

The reference words can be in any encoding scheme in which 
the bytes for whitespace, new lines, parenthesis, and dash (ascii 9, 10, 13, 32, 40,
41, and 45) always represent themselves.  This is true for UTF-8 and (I believe) 
EUC-JP, but not (I believe) for UTF-16 or GB18030-2000.  In addition, if the
encoding scheme contains multiple byte sequences that code for the same character,
then the reference and hypothesis words should both be normalized into an
encoding subset for which every character has an unique byte sequence.

=back

=head1 CTM FILE FORMAT

CTM (Conversation Time Marked) files are text files, any line of which may a
blank line, a comment line, or a regular line.  As with STM files, comment lines
begin with a semicolon (;) character, and blank and comment lines are ignored.

Regular CTM file lines give the information for a single hypothesis word, and
consist of either five or six whitespace separated fields:

=over 4

=item Field 1:  

Audio file identifier.  As in the STM file.

=item Field 2:  

Channel identifier.  As in the STM file.

=item Field 3:  

Word start time in seconds, as counted from the beginning of
the audio file.  Typically specified to 1/100ths of a second.

=item Field 4:  

Word duration in seconds.

=item Field 5:  

The hypothesis word.

=item Field 6 (Optional):  

A confidence score for the hypothesis word.  The
score must be between 0 and 1 inclusive.

=back

A CTM file may also contain alternate hypothesis paths.  These are typically
the result of filtering an initial CTM file with a GLM mapping file, and
are intended to deal with hypothesis words that have multiple valid transcriptions.
Alternate hypothesis pathes are described by a format that looks like

   fsh_109487 A 90.500 0.210 WHAT 0.645777
   fsh_109487 A * * <ALT_BEGIN>
   fsh_109487 A 90.710 0.230 THAT'S 0.347474
   fsh_109487 A * * <ALT>
   fsh_109487 A 90.710 0.115 THAT 0.347474
   fsh_109487 A 90.825 0.115 IS 0.347474
   fsh_109487 A * * <ALT>
   fsh_109487 A 90.710 0.115 THAT 0.347474
   fsh_109487 A 90.825 0.115 HAS 0.347474
   fsh_109487 A * * <ALT_END>
   fsh_109487 A 94.240 0.320 JUST 0.884898

Specifically, the alternate paths should be surrounded by the tokens
<ALT_BEGIN> and <ALT_END> in the CTM file word field, and the alternate
paths should be separated by <ALT>s, also in the word field.  In all of
these cases, fields 3 and 4 should contain only single asterisks.

For a particular audio file/channel combination, the words in a CTM file must
appear in order of increasing start time.  The UNIX command
"sort +0 -1 +1 -2 +2nb -3" will accomplish this while also sorting the 
conversations into an order sclite likes, but only if the CTM file does not
contain <ALT> regions.

=head1 OUTPUT

scorer.pl outputs four files:  the .sys, .raw, .sgml and .pra files.  These are
written to output-filename-prefix plus the suffix; if output-filename-prefix
is not given, ctm-hypothesis-file is used as the output filename prefix
instead.  The .sys file is additionally written to standard output.

=head2 The .sys File

The .sys file contains the following statistics for every label, every speaker,
and for the entire test set ("ALL"):

=over 4

=item #Ref = number of words in the reference STM file

=item #Hyp = number of words in the hypothesis CTM file

=item WER = the word error rate = ( #_substitutions + #_deletions + #_insertions ) / #_reference_words

=item %Cor = percentage correct = #_correct / #_reference_words

=item %Sub = percentage substitutions = #_substitutions / #_reference_words

=item %Del = percentage deletions = #_deletions / #_reference_words

=item %Ins = percentage insertions = #_insertions / #_reference_words

=item NCE = Normalized Cross Entropy, a measure of the goodness of the confidence
values in the CTM file.  It is calculated using the following formula:

  NCE = 1 - LL / ( #Cor log(p_c) + (#Hyp - #Cor) log(1-p_c) )
  LL = Log Likelihood of Confidence Values 
     = sum_{w correct} log( conf(w) ) + sum_{w incorrect} log( 1 - conf(w) )
  p_c = (ML Estimate of) Probability of Correctness
      = #Cor / #Hyp

In all of the above formulas, log( 0 ) is replaced with -1000 whenever it
occurs.

=back

=head2 The .raw File

The .raw file contains the following statistics for every label, every speaker,
and for the entire test set ("ALL"):

=over 4

=item #Ref = number of words in the reference STM file

=item #Hyp = number of words in the hypothesis CTM file

=item #Err = number of errors = #_substitutions + #_deletions + #_insertions

=item #Cor = number of correct hypothesis words

=item #Sub = number of substitutions

=item #Del = number of reference words deleted

=item #Ins = number of hypothesis words inserted

=item NCE = Normalized Cross Entropy, see above description under L<"The .sys File">

=back

=head2 The .pra File

The .pra file contains alignment information for each STM file reference
utterance.  Here is an example:

  Speaker fsh_110103_A Start time 123.44  End time 127.61
  Ref: (%HESITATION) TOPIC IS NEEDED OR OR WHERE THEY HAVE
  Hyp: OUR TOPIC IS NEEDED OR WHAT THEY HAVE
  Scores: ( #C #S #D #I ) = ( 7 1 1 1 )
  INSERTION: OUR
  CORRECT (Opt. Del.): (%HESITATION)
  CORRECT: TOPIC
  CORRECT: IS
  CORRECT: NEEDED
  DELETION: OR
  CORRECT: OR
  SUBSTITION: hypothesis WHAT for reference WHERE
  CORRECT: THEY
  CORRECT: HAVE

The utterance start and end times are given in seconds.  #C, #S, #D, and #I stand for number of correct, substitution, deletion and insertion words, respectively.  "Opt. Del." stands for optionally deletable (see L<"STM FILE FORMAT"> above).

Note that scorer.pl's .pra file output format is rather different than sclite's.

=head2 The .sgml File

The .sgml file also contains alignment information, but in a slightly
more computer parseable format.  Here is an example:

   <SYSTEM title="18985m-ADEC_CONF_SCORE-cleaned.ctm.filt" ref_fname="18985m-local-copy.stm.dedash.filt" hyp_fname="18985m-ADEC_CONF_SCORE-cleaned.ctm.filt" creation_date="Mon Feb 13 12:59:19 EST 2006" format="2.4" frag_corr="TRUE" opt_del="TRUE" weight_ali="FALSE" weight_filename="">
   <SPEAKER id="fsh_109487_A">
   <PATH id="(fsh_109487_A-86.06-88.02)" word_cnt="1" labels="<O>" file ="fsh_109487" channel="A" sequence="5" R_T1="86.06" R_T2="88.02" word_aux="h_t1+t2,h_conf">
   I,,"YEAH",86.18+86.62,0.911505:C,"YEAH","YEAH",87.31+87.74,0.943606
   </PATH>
   <PATH id="(fsh_109487_A-226.81-227.69)" word_cnt="1" labels="<O>" file ="fsh_109487" channel="A" sequence="40" R_T1="226.81" R_T2="227.69" word_aux="h_t1+t2,h_conf">
   S,"YUP","YEAH",227.04+227.42,0.940359
   </PATH>
   </SPEAKER>
   </SYSTEM>

The alignment information for each reference utterance is described by a
colon delimmited list.  Each alignment step is described by either

   C,"ref_word","hyp_word",start_time+end_time,confidence     [CORRECT]
   S,"ref_word","hyp_word",start_time+end_time,confidence     [SUBSTITUTION]
   I,,"hyp_word",start_time+end_time,confidence               [INSERTION]
   D,"ref_word",,,                                            [DELETION]

scorer.pl's .sgml file output is intended to be 100% compatible with sclite's, 
with the one exception of PATH id names:  scorer.pl's are
   wavefile_channel-starttime-endtime
while sclite's are
   wavefile_channel-number

=head1 ADVANTAGES OVER SCLITE

Better error messages.

Less finicky about input:  conversations do not need to appear in any particular
order, and it's okay if there are no hypothesis words for a speaker.

Totally case sensitive:  scorer.pl never upper or lower cases anything.

Fewer "special" characters:  words can now contain semicolons (;) and less than
signs (<), for example.

Small, easy to maintain implementation.

Summary statistics per label are put in the .sys file, rather than hidden
in separate .lur files.

=head1 CAVEATS

Some of sclite's output files aren't supported:  .det and .hist plots and
.lur files.  No sentence/utterance statistics or median statistics are output.

"IGNORE_TIME_SEGMENT_IN_SCORING" segments are properly ignored for scoring, but
they produce no alignment information.  (The effected hypothesis words should
be given "IGNORED" alignment tags in the .pra and .sgml files, but aren't.)

If there isn't at least a 2-3 second gap between two reference utterances,
they should be joined together for the purposes of aligning the hypothesis
words (and then separated again when outputing the alignment statistics).

Arguably, %Ins would make more sense as #_insertions / #_hypothesis_words,
but tradition and consistency define it as #_insertions / #_reference_words.

Nested <ALT> regions in CTM files and multiple reference paths in STM files
are not supported.

There should be command line options to fiddle with various things
(insertion/deletion/substitution costs, whether to match word fragments,
which output files to produce, etc.).

=head1 AUTHOR

Thomas Colthurst, thomasc@bbn.com.  Z<BBN_ref_explicitly_OK>

=head1 COPYRIGHT

Copyright 2005 by BBN Technologies.

