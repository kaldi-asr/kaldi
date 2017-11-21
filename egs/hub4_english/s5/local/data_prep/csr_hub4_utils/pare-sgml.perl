#!/usr/bin/perl

# $Id: pare-sgml.perl,v 1.3 1996/08/15 02:51:17 robertm Rel $
# removes extraneous headers and other non-LM fields
# translates <DOC ...> into LM-standard <art ...>
# removes comments (enclosed in brackets)

use strict;
use warnings;

my $intext=0;
while (<>)
{
    if ($intext == 0)
    {
	print if (s=<DOC=<art=);
	print if (s=</DOC=</art=);
	$intext = 1 if /^<TEXT>/;
	next;
    }
    if (/^<\/TEXT>/)
    {
	$intext = 0;
	next;
    }
    next if /^<comment>/;
    next if /^<speaker>/;

    s/\[+[^\[\]]*\]+//g;
    if (/[\[\]]/)
    {
	warn "pare-sgml: warning - unbalanced comment brackets at $ARGV line $.\n";
	print STDERR " line=$_";
    }
    print;
}
