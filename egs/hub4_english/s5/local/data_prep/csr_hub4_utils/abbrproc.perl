#!/usr/bin/perl
# $Id: abbrproc.perl,v 1.3 1996/08/21 20:05:09 robertm Rel $
###############################################################################
# This software is being provided to you, the LICENSEE, by the Massachusetts  #
# Institute of Technology (M.I.T.) under the following license.  By           #
# obtaining, using and/or copying this software, you agree that you have      #
# read, understood, and will comply with these terms and conditions:          #
#                                                                             #
# Permission to use, copy, modify and distribute, including the right to      #
# grant others the right to distribute at any tier, this software and its     #
# documentation for any purpose and without fee or royalty is hereby granted, #
# provided that you agree to comply with the following copyright notice and   #
# statements, including the disclaimer, and that the same appear on ALL       #
# copies of the software and documentation, including modifications that you  #
# make for internal use or for distribution:                                  #
#                                                                             #
# Copyright 1991-4 by the Massachusetts Institute of Technology.  All rights  #
# reserved.                                                                   #
#                                                                             #
# THIS SOFTWARE IS PROVIDED "AS IS", AND M.I.T. MAKES NO REPRESENTATIONS OR   #
# WARRANTIES, EXPRESS OR IMPLIED.  By way of example, but not limitation,     #
# M.I.T. MAKES NO REPRESENTATIONS OR WARRANTIES OF MERCHANTABILITY OR FITNESS #
# FOR ANY PARTICULAR PURPOSE OR THAT THE USE OF THE LICENSED SOFTWARE OR      #
# DOCUMENTATION WILL NOT INFRINGE ANY THIRD PARTY PATENTS, COPYRIGHTS,        #
# TRADEMARKS OR OTHER RIGHTS.                                                 #
#                                                                             #
# The name of the Massachusetts Institute of Technology or M.I.T. may NOT be  #
# used in advertising or publicity pertaining to distribution of the          #
# software.  Title to copyright in this software and any associated           #
# documentation shall at all times remain with M.I.T., and USER agrees to     #
# preserve same.                                                              #
###############################################################################

# abbreviation preprocessor for WSJ
# assumes 1 sentence per line
#
# 1. map "x.y." -> "x. y."
# 2. convert Roman numerals with appropriate left context into cardinal no.s
# 3. expand abbreviations and word translations
#	expands remaining Roman numerals into ordinal no.s
# 4. map isolated letters: "x" -> "x."

# Minor modifications by David Graff, Linguistic Data Consortium, in
# preparation for publishing on cdrom;  Aug. 11, 1994.

# Major modifications by Robert MacIntyre, LDC, attempting to improve
# performance (~50% speedup), in preparation of Broadcast News material,
# August 1996.


$file="$ENV{HOME}/bc-news/bin/abbrlist";		# default abbreviation file

for($i=0,$j=0;$i<=$#ARGV;$i++)
{	if($ARGV[$i] =~ /^-/)
	{	if($ARGV[$i] =~ /^-v/) {$vflg=1;}
		else {&perr("illegal flag: $ARGV[$i]");}
	}
	else
	{ #	if($file) {&perr("multiple file arg");}
		$file=$ARGV[i];
	}
}
@ARGV=();
if(!file) {&perr("no abbreviation file specified"); }

if(!open(FILE,$file)) {&perr("cannot open abbreviation file"); }
while(<FILE>)
{	if(/^#/) {next;}	# comment
	s/\n//;
	if(!$_) {next;}		# blank
	$y=$_;
	s/^(\S+)\s+//;		# extract 1st word
	$x=$1;
	if(!$x) {&perr("no word: $y");}
	if(!$_) {&perr("no value: $y");}

	if($x =~ /^\*r/)		# left context for roman numeral
	{	if(!/^[a-zA-Z]{2,}$/)
			{&perr("illegal roman: $x");}
		tr/a-z/A-Z/;		# map to UC
		$romanlc{$_}=1;
	}
	elsif($x =~ /\.$/)			# abbreviations
	{	if($x !~ /^[a-zA-Z][a-zA-Z\.]+\.$/)
			{&perr("illegal abbreviation: $x");}
		$x =~ s/\.$//;
		$abbrev{$x}=$_;
		if($x =~ /[a-z]/)
		{	$x =~ tr/a-z/A-Z/;	#UC version
			tr/a-z/A-Z/;
			$abbrev{$x}=$_;
		}
		#if(length($x)>$maxabl) {$maxabl=length($x);}
	}
	else				# translations
	{	if($x !~ /^[a-zA-Z\.&\/-]+[a-zA-Z]$/)
			{&perr("illegal translation: $x");}
		$trans{$x}=$_;
		if($x =~ /[a-z]/)
		{	$x =~ tr/a-z/A-Z/;	#UC version
			tr/a-z/A-Z/;
			$trans{$x}=$_;
		}
		#if(length($x)>$maxtrl) {$maxtrl=length($x);}
	}
	$n++;
}
#if($vflg) {print STDERR "$n lines read from file\n";}

&setupRoman;

while(<>)
{ ###########################  abbrevproc ####################################

    # pass SGML as is
    if (/^<\/?[spa]/)
    {
	print;
	next;
    }
    chop;


    s/&/ & /g;			# &
    s=/= / =g;			# /
    s/ - / -- /g;		# save (long) dashes
    s/\b(-+)\b/ $1 /g;		# -, --, etc. in words
    s/([^-\s])(-+)([^-\s])/$1 $2 $3/g;

    if(/_/)
    {
	&perr2("removing illegal underscores (_) in:\n $_\n");
	s/_//g;
    }

    @input = split(/\s+/);
    @output=();
    for($field=0;$field<=$#input;$field++)
    {
	$_ = $input[$field];
	# if($vflg) {print "in: $_\n";}

	s/^(\W*)//;		# strip front
	$front=$1;

	s/(\W*)$//;		# strip back
	$back=$1;
	if(/\.?\'[sS]$/)		# possessive
	{
	    s/(\.?\'[sS])$//;
	    $back="$1$back";
	}
	elsif (/^[A-Z]+s$/)	# eg Cs or Xs
	{
	    s/s$//;
	    $back="_s$back";
	}

	$ptbkflg = ($back =~ /^\./);

	#if($vflg) {print "f=$front, m=$_, b=$back\n";}


	# Roman numerals
	if(/^[IVX]{1,6}$/ && $front eq "" && $field>0 &&
	   ($x=&geto()))
	{
	    $x =~ tr/a-z/A-Z/;	# map lc to UC
	    $x =~ s/^\W//;	   # strip initial punct from lc
	    if($romanlc{$x})	# left context check
	    {
		if($front) 
		{
		    &pusho($front);
		    if($front !~ /[\w]$/) {$appendflg=1;}
		}

		if ($x=$Roman{$_})
		{
		    &pusho($x);
		}
		else
		{
		    &perr2("illegal roman: $_");
		    &pusho($_);
		}

		if($back)
		{
		    if($back !~ /^[\w]/) {&appendo($back);}
		    else {&pusho($back);}
		}
		next;
	    }
				
	}


	# St. or St ["Street" vs. "Saint"]
			if($_ eq "St")
			{	$back =~ s/^\.//;
				if($front ne "" && $back ne "")
				{	&perr2("Cannot resove St.: $input[$field-1] $input[$field] $input[$field+1]");
					$x=Street;	# Wild guess
				}
				elsif($front) { $x="Saint"; }
				elsif($back) { $x="Street"; }
				elsif($input[$field-1] !~ /^[A-Z]/
					&& $input[$field+1] =~ /^[A-Z]/)
					{ $x = "Saint"; }
				elsif($input[$field-1] =~ /^[A-Z]/
					&& $input[$field+1] !~ /^[A-Z]/)
					{ $x = "Street"; }

				elsif(!$back && $input[$field+1] =~ /^[A-Z]/)
					{ $x = "Saint"; }
				elsif(!$back && $input[$field+1] eq '-' &&
					$input[$field+2] =~ /^[A-Z]/)
					{ $x = "Saint"; }
				else
				{	&perr2("Cannot resove St.: $input[$field-1] $input[$field] $input[$field+1]");
					$x=Street;	# Wild guess
				}


				if($front) 
				{	&pusho($front);
					if($front !~ /[\w]$/) {$appendflg=1;}
				}
	
				&pusho($x);

				if($back)
				{	if($back !~ /^[\w]/) {&appendo($back);}
					else {&pusho($back);}
				}
				next;
			}

	# abbreviations (end with .)
			if($ptbkflg && ($x=$abbrev{$_}))
			{	
					if($front) 
					{	&pusho($front);
						if($front !~ /[\w]$/)
							{$appendflg=1;}
					}
	
					&pusho($x);
					
					if($field<$#input || $back =~ /[!?]/)
						{ $back =~ s/^\.//; }	# rm .
					else			# end of sent
					{	$back =~ s/^\.(\'s)/$1./;
						if($back =~ /\..*\./) # 2 dots
						      {$back=~s/\.([^\.]*)/$1/;}
					}

					if($back)
					{	if($back !~ /^[\w]/)
							{&appendo($back);}
						else {&pusho($back);}
					}
					next;
				
			}

	# translations (do not end with .)
			# first merge multi-token translations
			if($input[$field+1] =~ /^[-\/&]$/ && $back eq "")
			{	$x=$input[$field+2];
				$x =~ s/(\W*)$//;
				$xback=$1;
				if($x =~ /\.?\'[sS]$/)		# possessive
				{	$x =~ s/(\.?\'[sS])$//;
					$xback="$1$xback";
				}
				elsif ($x =~ /^[A-Z]+s$/)	# eg Cs or Xs
				{	$x =~ s/s$//;
					$xback="_s$xback";
				}
				if($trans{"$_$input[$field+1]$x"})   # eg. AT&T
				{	$_="$_$input[$field+1]$x";
					$field+=2;

					$back=$xback;
					$ptbkflg = ($back =~ /^\./);
				}
			}
			# then see if we have a translation
			if ($x=$trans{$_})
			{	if($front)
				{	&pusho($front);
					if($front !~ /[\w]$/) {$appendflg=1;}
				}
	
				&pusho($x);
					
				if($x =~ /\.$/) { $back =~ s/^\.//; } # only 1 .
				if($back)
				{	if($back !~ /^[\w]/) {&appendo($back);}
					else {&pusho($back);}
				}
				next;
			}

	# eg. Cs, but not As Is Ms Us
			if(($back =~ /^_s/) && /^[B-HJ-LN-TV-Z]$/)  
			{	if($front)
				{	&pusho($front);
					if($front !~ /[\w]$/) {$appendflg=1;}
				}
	
				&pusho("$_.");
	
				if($back)
				{	if($back !~ /^[\w]/) {&appendo($back);}
					else {&pusho($back);}
				}
				next;
			}

	# split x.y.
	$_ .= '.' if $ptbkflg;	# NOTE THIS CHANGES $_ FOR FUTURE MATCHES
				# but it has no more uses in this loop,
				# so this _should_ be okay.
	if (/^([a-zA-Z]\.)+([sS]?)$/)
	{
	    $sflag = $2;	# remember if plural (as opposed to a.s.)

	    chop if $ptbkflg;	# trim period that we just added

	    s/\./. /g;		# x.y. -> x. y.

	    s/ ([sS])$/$1/ if $sflag;	# reattach final "s"

	    if($front) 
	    {	&pusho($front);
		if($front !~ /[\w]$/) {$appendflg=1;}
	    }
	
	    &pusho($_);

	    if($back)
	    {	if($back !~ /^[\w]/) {&appendo($back);}
		else {&pusho($back);}
	    }
	    next;
	}

	# remaining tokens are passed "as is"
	# [Below does "&pusho($input[$field]);" but faster, since we avoid
	# the subroutine call for the most common case.]
	push(@output,$input[$field]);
    }

    $_=join(" ",@output);

    # if($vflg) {print "ab:\t$_\n";}

    #########################  lettproc  ######################################
    if (/\b[b-zB-HJ-Z]\b/)
    {
	@output = split(/\s+/);

	foreach(@output)
	{
	    next unless /^\W*[b-zB-HJ-Z]\W*$/;

	    #if($vflg) {print "le: $_\n";}

	    # some cases to skip/pre-change.  (Note that backslashing of
	    # quotes is for the sake of Emacs, not Perl.)
	    next if (/^[\'][nN]$/);		# Spic \'n Span

	    s/(^[\`\'][nN])[\`\']$/$1/ && next;	# Rock 'n' Roll: 'n' -> \'n

	    s/^[\`\'\"]R[\'\`\"]$/"R"/ && next;	# Toys "R" Us

	    next if (/^o\'$/);			# Man o\' War

	    # put . at end of remaining single-letter words
	    s/^(\W*)([b-zB-HJ-Z])([^.\w]\W*|[^\w.]*)$/$1$2.$3/;
	}
	
	$_=join(" ",@output);
    }

    s/\s+/ /g;
    s/^ //;
    s/ $//;

    s/ _//g;	# attach final s for Cs or AFLs
    s/_//g;	# clear _
    s/ - /-/g;

    print $_,"\n" if $_;
}

sub pusho				# pusho($x): push output
{	if($appendflg)			# global: used for fronts
	{
		&appendo(@_[0]);
	}
	else {push(@output,@_);}
}

sub appendo				# appendo($x): append to output
{	$appendflg=0;		
	if($#output < 0) {&perr("appendo: output empty");}
	$output[$#output] .= @_[0];
}

sub geto				# geto(): get last output
{	if($#output < 0) {print STDERR ("geto: output empty\n");}
	return $output[$#output];
}

sub perr
{	print STDERR "abbrevproc: $_[0]\n";
	exit(1);
}

sub perr2
{	print STDERR "abbrevproc: $_[0]\n";
}

sub setupRoman
{
    $Roman{I}="one";
    $Roman{II}="two";
    $Roman{III}="three";
    $Roman{IV}="four";
    $Roman{V}="five";
    $Roman{VI}="six";
    $Roman{VII}="seven";
    $Roman{VIII}="eight";
    $Roman{IX}="nine";
    $Roman{X}="ten";
    $Roman{XI}="eleven";
    $Roman{XII}="twelve";
    $Roman{XIII}="thirteen";
    $Roman{XIV}="fourteen";
    $Roman{XV}="fifteen";
    $Roman{XVI}="sixteen";
    $Roman{XVII}="seventeen";
    $Roman{XVIII}="eighteen";
    $Roman{XIX}="nineteen";
    $Roman{XX}="twenty";
    $Roman{XXI}="twenty-one";
    $Roman{XXII}="twenty-two";
    $Roman{XXIII}="twenty-three";
    $Roman{XXIV}="twenty-four";
    $Roman{XXV}="twenty-five";
    $Roman{XXVI}="twenty-six";
    $Roman{XXVII}="twenty-seven";
    $Roman{XXVIII}="twenty-eight";
    $Roman{XXIX}="twenty-nine";
    $Roman{XXX}="thirty";
    $Roman{XXXI}="thirty-one";
    $Roman{XXXII}="thirty-two";
    $Roman{XXXIII}="thirty-three";
    $Roman{XXXIV}="thirty-four";
    $Roman{XXXV}="thirty-five";
}
