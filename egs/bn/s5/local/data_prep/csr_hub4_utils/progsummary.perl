#!/usr/bin/perl

# Program:	progsummary.perl
# Written by:	dave graff
# Usage:	[file.list]
# Purpose:	extracts program information from sgml-ized PSM texts

$degbug = 0;
if ( $ARGV[0] eq "-d" ) {
    $debug = 1;
    shift;
}

while (<>)
{
    chop;
    open( INP, "<$_" );
    $progdate = $progid = "unknown";
    while (<INP>) {
	if ( /^<program>/ ) {
	    $_ = <INP>;
	    print STDERR if ( $debug );
	    $netwrk = substr( $_, 0, 3 );
	    $rest = substr( $_, 3 );
	    if ( $rest =~ /^(20\/20)/ ) {
		$progid = $1;
	    }
	    elsif ( $rest =~ /^([A-Z a-z\&]+)/ ) {
		$progid = $1;
	    }
	}
	elsif ( /^<summary>/ ) {
	    $_ = <INP>;
	    print STDERR "$_===\n" if ( $debug );
	    if ( /\d+\\(\d{6})\\\d+/ ) {
		$progdate = $1;
	    }
	}
	elsif ( /^<\/art>/ ) {
	    print "$netwrk\t$progdate\t\"$progid\"\n";
	}
    }
    close INP;
}
