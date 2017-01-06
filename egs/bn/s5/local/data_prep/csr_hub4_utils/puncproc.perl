#!/usr/bin/perl

# $Id: puncproc.perl,v 1.2 1996/08/05 16:12:42 robertm Rel $
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

# punctuation preprocessor for WSJ 
# assumes 1 sentence per line
# places spaces around punctuation and translates to IBM-like notation
#
# punctproc -np removes punctuation
#
# NOTE: wsj89 starts single quotes with ` or '
#

for($i=0,$j=0;$i<=$#ARGV;$i++)
{	if($ARGV[$i] =~ /^-/)
	{	if($ARGV[$i] =~ /^-np$/) {$npflg=1;}
		else {&perr2("illegal flag: $ARGV[$i]");}
	}
	else { &perr2("no file args"); }
}
@ARGV=();

while(<>)
{	s/^/ /;
	s/\n$/ /;

	next if (/<\/?[spa]/);	# protect sgml

						# forbidden symbols
	if(/</) {&perr("<");}				# <
	if(/>/) {&perr(">");}				# >
	if(/\$/) {&perr("\$");}				# $
	if(/_/) {&perr("_");}				# _
	if(/\d/) {&perr("[0-9]");}			# 0-9

					# protect contractions with _
	s/([a-zA-Z]in')([^a-zA-Z])/$1_ $2/g; 	# *in'	e.g. Dunkin', singin'
						# Rock 'n' Roll
	s/(\W)['`]([nN])(\W)/$1 _'$2$3/g;	# [`'][nN] ->  _'[nN]
	s/(\W)([nN]')(\W)/$1$2_ $3/g;		# [nN]'
	s/(\W)('[eE]m)(\W)/$1_$2$3/g;		# '[eE]m
	s/(\W)[`'"]R\.?['"](\W)/$1 _"R."_ $3/g;	# toys "R" us
	s/(\W)(Cos.')(\W)/$1$2_ $3/g;		# Cos.'	(companies')
	s/(\W)(de.')(\W)/$1$2_ $3/g;		# de'	 Imelda de' Lambertazzi
	s/(\W)(Bros.')(\W)/$1$2_ $3/g;		# Bros.'
	s/(\W)(o')(\W)/$1$2_ $3/g;		# o'	 Man o' War
	s/(\W)(ol')(\W)/$1$2_ $3/g;		# ol'	 old
	s/(\W)maitre *d'(\W)/$1maitre_d'_ $2/g;	# maitre d'
	s/(\W)maitres *d'(\W)/$1maitres_d'_ $2/g; # maitres d'
	s/(\W)('neath)(\W)/$1 _$2$3/g;		# 'neath	 beneath
	s/(\W)('Wadoo)(\W)/$1 _$2$3/g;
			# 'Wadoo   'Wadoo , zim bam , boodleoo , hoodle ahdawam
	s/(\W)('cause)(\W)/$1 _$2$3/g;		 # 'cause	because
	s/(\W)('burbs)(\W)/$1 _$2$3/g;		# 'burbs	suburbs
	s/(\W)('[nN]uf)(\W)/$1 _$2$3/g;		# 'Nuf	enough
	s/(\W)('til)(\W)/$1 _$2$3/g;		# 'til


	s/([^\w\.\'\`_ -])/ $1 /g;		# SP around most punct 
						# but not .'`\_-

	if(!$npflg)
	{	s/ty-(one)/ty $1/g;		# rm - from twenty-one
		s/ty-(first)/ty $1/g;		# rm - from twenty-first
		s/ty-(two)/ty $1/g;		# rm - from twenty-two
		s/ty-(second)/ty $1/g;		# rm - from twenty-second
		s/ty-(three)/ty $1/g;		# rm - from twenty-three
		s/ty-(third)/ty $1/g;		# rm - from twenty-third
		s/ty-(four)/ty $1/g;		# rm - from twenty-four
		s/ty-(five)/ty $1/g;		# rm - from twenty-five
		s/ty-(six)/ty $1/g;		# rm - from twenty-six
		s/ty-(seven)/ty $1/g;		# rm - from twenty-seven
		s/ty-(eight)/ty $1/g;		# rm - from twenty-eight
		s/ty-(nin)/ty $1/g;		# rm - from twenty-nine{th}
	}
	#s/([^-])-([^-])/$1 - $2/g;		# -
	#s/([^-])-([^-])/$1 - $2/g;		# -

	s/([^\.]) *\. *\. *\. *\. *([^\.])/$1 _..._ . $2/g;	# x ... .
	s/([^\.]) *\. *\. *\. *([^\.])/$1 _..._ $2/g;	# x ...  

	s/([^\w'\.][b-zB-HJ-Z]\.)([^\.\w]*)$/$1 .$2/;	# eg. S. at end -> S. .
	s/(\s[a-z]\.\s[a-z]\.)([^\.\w]*)$/$1 .$2/i; #eg. S. I. at end -> S. I. .
	s/(\WMr\.)(\W*)$/$1 . $2/i;		# Mr. at end -> Mr. .
	s/(\WMrs\.)(\W*)$/$1 . $2/i;		# Mrs. at end -> Mrs. .
	s/(\WMs\.)(\W*)$/$1 . $2/i;		# Ms. at end -> Ms. .
	s/(\WMessrs\.)(\W*)$/$1 . $2/i;		# Messrs. at end -> Messrs. .

	s/\.([^.\w]*)$/ . $1/;			# SP around . at end of sent

	s/([^\w\.])['`]([a-zA-Z]*)'(\W)/$1 ' $2 ' $3/g;	# `word'   
	s/([^\w\.])['`]([a-zA-Z])/$1 ' $2/g;			# 'word
	s/([^sS])' /$1 ' /g;			# non plural-possessives

	s/([^_])`/$1 ` /g;			# SP around ` (should not need)
	s/`/'/g;				# ` -> '      (should not need)

	s/_/ /g;				# clear _

	if(!$npflg)
	{	s/ , / ,COMMA /g;			# map punct to words
		s/ \? / ?QUESTION-MARK /g;
		s/ : / :COLON /g;
		s/ # / #SHARP-SIGN /g;
		s/ @ / @AT-SIGN /g;
		s/ ' / 'SINGLE-QUOTE /g;
		s/ " / "DOUBLE-QUOTE /g;
		s/ ; / ;SEMI-COLON /g;
		s/ ! / !EXCLAMATION-POINT /g;
		s/ & / &AMPERSAND /g;
		s/ \+ / +PLUS /g;
		s/ \{ / {LEFT-BRACE /g;
		s/ \} / }RIGHT-BRACE /g;
		s/ \( / (LEFT-PAREN /g;
		s/ \) / )RIGHT-PAREN /g;
		s/ \. / .PERIOD /g;
		s/ \.{3} / ...ELLIPSIS /g;
		s/ -- / --DASH /g;
		# s/ - / -HYPHEN /g;
		s/ = / =EQUALS /g;
		s/ % / %PERCENT /g;
		s/ \/ / \/SLASH /g;
		s/ ([b-zB-HJ-Z]) / $1. /g;   # restore . removed by elipsis err
	}
	else
	{	s/ , / /g;			# map punct to words
		s/ \? / /g;
		s/ : / /g;
		s/ # / /g;
		s/ @ / at /g;
		s/ ' / /g;
		s/ " / /g;
		s/ ; / /g;
		s/ ! / /g;
		s/ & / and /g;
		s/ \+ / plus /g;
		s/ \{ / /g;
		s/ \} / /g;
		s/ \( / /g;
		s/ \) / /g;
		s/ \. / /g;
		s/ \.{3} / /g;
		s/ -- / /g;
		s/ ?- ?/ /g;
		s/ = / equals /g;
		s/ % / percent /g;
		s/ \/ / slash /g;
		s/\.POINT/point/g;
	}
} continue {
	# this block is executed even if we use "next"
	s/ {2,}/ /g;
	s/^ //;
	s/ $//;
	if($_) {print "$_\n";}
}

sub perr		#perr(error,line);
{	print STDERR "punctproc: line no=$.: $_[0]\n";
	print STDERR "line=$_\n";
}

sub perr2
{	print STDERR "num: $_[0]\n";
	exit(1);
}
