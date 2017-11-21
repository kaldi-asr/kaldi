#!/usr/bin/perl

# artfilter.perl 

# This perl script can be used to (de)select articles from TIPSTER
# format newswire data on the basis of the content of a specific
# tagged element.  This version allows a number of string patterns
# (drawn from a separate input file) to be checked against the content
# of a chosen tag, and allows residue articles to be sent to a
# separate file (in addition to having selected articles written to
# stdout).

require "newgetopt.pl";
$cmd_okay = &NGetOpt( 't=s', 'p=s', 'f=s', 'r=s', 'v', 'x' );
$arg_okay = ( $opt_t ne "" && ( $opt_p ne "" || $opt_f ne "" ));

if ( ! $cmd_okay || ! $arg_okay ) {
    print 
"\nUsage: artfilter.perl -t tag [-p ptrn | -f ptrns] [-r resid] [-vx] [infile]\n";
    print "  writes DOCs with <tag> containing /ptrn(s)/ to stdout\n";
    print "  -v = select DOCs NOT containing /ptrn(s)/ in <tag>\n";
    print "  -x = exclude DOCs that do not contain <tag>\n";
    print "  -r = write residue DOCs to resid file\n";
    exit;
}

@patrns = ();
if ( $opt_f ne "" ) {
    open( PATRNS, "<$opt_f" );
    while (<PATRNS>) {
	chop;
	push( @patrns, $_ );
    }
} else {
    push( @patrns, $opt_p );
}
close PATRNS;

if ( $opt_r ) {
    open( RESID, ">$opt_r" );
}

$outputOn = $foundtag = 0;

while (<>) 
{
    if ( /<DOC[ >]/ ) {
	$artbuf = $_;
	$outputOn = 1;
    }
    elsif ( /<\/DOC>/ ) {
	if ( $outputOn ) {
	    $artbuf .= $_;
	    if ( $outputOn == 1 && ( ! $opt_x || $foundtag )) {
		print $artbuf;
	    } elsif ( $opt_r && ( ! $opt_x || $foundtag )) {
		print RESID $artbuf;
	    }
	    $outputOn = 0;
	}
	$foundtag = 0;
    }
    elsif ( $outputOn ) {
	$artbuf .= $_;
	if ( /\<$opt_t/ ) {
	    $foundtag = 1;
	    $tagdata = $_;
	    while ( $tagdata !~ /\<\/$opt_t/ ) {
		$_ = <>;
		$artbuf .= $_;
		$tagdata .= $_;
	    }
	    foreach $ptn ( @patrns ) {
		last if (( $i = ( $tagdata !~ /$ptn/ )) == 0 );
	    }
	    if ( $i ^ $opt_v ) { $outputOn = ( $opt_r ) ?  2 : 0; }
	}
    }
}

if ( $opt_r ) {
    close RESID;
}
