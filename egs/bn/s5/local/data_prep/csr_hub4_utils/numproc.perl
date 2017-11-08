#! /usr/bin/perl
#
# $Id: numproc.perl,v 1.7 1996/08/23 05:04:11 robertm Rel $
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

# preprocessor for WSJ
# assumes 1 sentence per line
#
# 1.  expand numerical exceptions: eg. 386
# 2.  do regular numerical expansions

# Minor modifications by David Graff, Linguistic Data Consortium, in preparation
# for publishing on cdrom;  Aug. 11, 1994.

$POINT='.POINT';		# orthographic notation for .

	# final s in name indicates plural version, otherwise just add s
@ones_z=("zero","one","two","three","four",
	"five","six","seven","eight","nine");
@ones_oh=("oh","one","two","three","four",
	"five","six","seven","eight","nine");
@ten=("","ten","twenty","thirty","forty","fifty",
	"sixty","seventy","eighty","ninety");
@teen=("ten","eleven","twelve","thirteen","fourteen","fifteen",
	"sixteen","seventeen","eighteen","nineteen");
@mult=("","thousand","million","billion","trillion"
	,"quadrillion","quintillion","sextillion","septillion","octillion");
@den=("","","half","third","quarter","fifth",
	"sixth","seventh","eighth","ninth","tenth",
	"eleventh","twelfth","thirteenth","fourteenth","fifteenth",
	"sixteenth","seventeenth","eighteenth","nineteenth");
@largeden=("","first","second","third","fourth","fifth",
	"sixth","seventh","eighth","ninth","tenth",
	"eleventh","twelfth","thirteenth","fourteenth","fifteenth",
	"sixteenth","seventeenth","eighteenth","nineteenth");
@ordnal=("","first","second","third","fourth","fifth",
	"sixth","seventh","eighth","ninth","tenth",
	"eleventh","twelfth","thirteenth","fourteenth","fifteenth","sixteenth");
@months=("Jan.","Feb.","Mar.","Apr.","Jun.","Jul.","Aug.","Sept.","Oct.",
	"Nov.","Dec.","January","February","March","April","May","June",
	"July","August","September","October","November","December");

$exfile="$ENV{HOME}/bc-news/bin/num_excp";		# default exceptions file name

for($i=0,$j=0;$i<=$#ARGV;$i++)
{	if($ARGV[$i] =~ /^-/)
	{	if($ARGV[$i] =~ /^-v/) {$vflg=1;}
		elsif($ARGV[$i] =~ /^-x/)
		{	$exfile=$ARGV[$i];
			$exfile =~ s/^-x//;
		}
		else {&perr2("illegal flag: $ARGV[$i]");}
	}
	else { &perr2("no file args"); }
}
@ARGV=();

if(!exfile) {&perr2("no exceptions file specified"); }

if(!open(EXFILE,$exfile)) {&perr2("cannot open $exfile"); }
while(<EXFILE>)
{	if(/^#/) {next;}	# comment
	s/\n//;
	if(!$_) {next;}		# blank
	$y=$_;
	s/^(\S+)\s*//;		# extract 1st word
	$x=$1;
	if($x eq "") {&perr2("$exfile: no word: $y");}
	if($x =~ /^\$\$/)		# $$word => skip
	{	$x =~ s/^\$*//;
		$sing_dollar{$x}=2;
	}
	elsif($x =~ /^\$/)		# $word => singular right context
	{	$x =~ s/^\$*//;
		$sing_dollar{$x}=1;
	}
	elsif($x =~ /^\*/)
	{	$x =~ s/\**//g;
		if(!$x) {&perr2("$exfile: no serno word");}
		$sernowd{$x}=1;		# serial no words
	}
	else
	{	if($x !~ /\d/) {&perr2("$exfile: non-numerical key");}
		if(!$_) {&perr2("$exfile: no value");}

		$except{$x}=$_;		# translations
	}
	$n++;
}
close(EXFILE);
if($vflg) {print STDERR "$n lines read from exceptions file\n";}

for($i=0;$i<=$#months;$i++)	# make months hash
{	$_=$months[$i];
	$months{$_}=1;		# mixed case
	tr/a-z/A-Z/;
	$months{$_}=1;		# UC
}

while(<>)
{	# removed local($front.$back,$x) to conserve memory RWM 8/96

##############################  exceptproc  ##################################
	s/^\s*//;
	s/\n//o;
	if($vflg) {print "input:\t$_\n";}
	if(/\d/ && !/^<\/?[spa]/)		# opt and protect sgml
	{	@input = split(/\s+/o);
		@output=();
		for($field=0;$field<=$#input;$field++)	# $field is global
		{	$_=$input[$field];
	
			if(!/\d/)			# only processes numbers
			{	&pusho($input[$field]);		# not processed
				next;
			}
	
			s/^(\W*)//o;		# strip front
			$front=$1;
			if($front =~ /\$$/ || $front =~ /#$/)	# protect money
			{	&pusho($input[$field]);		# not processed
				next;
			}
	
			s/(\W*)$//o;		# strip back
			$back=$1;
	
			if($front =~ /\'$/ && $except{"'$_"})	# eg "'20s"
			{	$front =~ s/\'$//;
				if($front) 
				{	&pusho($front);
					if($front !~ /[\w]$/o) {$appendflg=1;}
				}
	
				&pusho($except{"'$_"});		# translation
					
				if($back)
				{	if($back !~ /^[\w]/o) {&appendo($back);}
					else {&pusho($back);}
				}
			}
			elsif($except{$_})
			{	if($front) 
				{	&pusho($front);
					if($front !~ /[\w]$/o) {$appendflg=1;}
				}
	
				&pusho($except{$_});		# translation
					
				if($back)
				{	if($back !~ /^[\w]/o) {&appendo($back);}
					else {&pusho($back);}
				}
			}
			else {&pusho($input[$field]);}		# not processed
		}
		$_=join(" ",@output);
	}
	s/\s+/ /g;
	s/^ //o;
	s/ $//o;
	if($vflg) {print "ex:\t$_\n";}

############################  numproc  ########################################
	if(!/^<\/?[spa]/)			# protect sgml, also art
	{	s/(\d+)-(\d+)-(\d+)/$1 - $2 - $3/g;	# eg. 1-2-3
		s/(\d+)x(\d+)/$1 by $2/g;		# eg. 2x4
		s/(\d+)\+(\d+)/$1 plus $2/g;		# eg. 2+2
		s=(\d)-(\d)[/\\](\d)=$1 $2/$3=g;	# e.g. 3-1/2
		s=(\d)\\(\d)=$1/$2=g;			# e.g. 1\2 for 1/2
		s/\$(\d[\d,]*)-\$(\d)/$1 to \$$2/g;	# $ range: eg. $1-$2
		s/\$(\d[\d,]*)-(\d)/$1 to \$$2/g;	# $ range: eg. $1-2
		s/(\d)-(\'?)(\d)/$1 to $2$3/g;		# range: eg. 1-2
		s/%-(\d)/% to $1/g;			# % range: eg. 1%-2%
		s/(\d)=(\d)/$1 equals $2/g;		# equation: x=y
		s/ - / -- /g;				# recode dashes
		s/([^-\d\s])-([^-\d\s])/$1 - $2/g;	# split in-word hyphens
		s/- +-/--/g; s/- +-/--/g;		# close dashes
		s/-{3,}/--/g;				# map dashes to --
		s/--/ -- /g;				# space around --
		s/(\d) +(\d+\/\d)/$1 and $2/g;	      # dig frac -> dig and frac
		s/([a-zA-Z])\//$1 \/ /g;		# text/*
		s/\/([a-zA-Z])/ \/ $1/g;		# */text

		s/([a-zA-Z]\d+)\/(\d+)/$1 \/ $2/g;	# eg. a1/3 -> a1 / 3
		s/(\/\d*)th/$1/ig;			# eg. 1/10th -> 1/10
		s/(\/\d*1)st/$1/ig;			# eg. 1/x1st -> 1/x1
		s/(\/\d*2)nd/$1/ig;			# eg. 1/x2nd -> 1/x2
		s/(\/\d*3)rd/$1/ig;			# eg. 1/x3rd -> 1/x3
		s/(\d+)\/(\d+[a-zA-Z])/$1 \/ $2/g;	# eg. 1/3a -> 1 / 3a
		s/([a-zA-Z])-(19\d\d\D)/$1 - $2/g;  # eg. mid-1990 -> mid - 1990
#		s/([a-zA-Z])-(\d)/$1 $2/g;		# eg. a-1 -> a 1
#		s/(\d)-([a-zA-Z])/$1 $2/g;		# eg. 1-a -> 1 a
		s/([a-zA-Z])-(\d)/$1 - $2/g;		# eg. a-1 -> a - 1
		s/(\d)-([a-zA-Z])/$1 - $2/g;		# eg. 1-a -> 1 - a
	
		# fix common time typo (; for :)
		s/\b([012]?\d);([0-5]\d)\b/$1:$2/g;	# e.g. 11;00 -> 11:00

		if(!/\d:\d\d$/o && !/\d:\d\d\D/o)    # preprocess non-time \d:\d
		{	s/(\d):(\d)/$1 : $2/g;
			s/(\S):(\d)/$1: $2/g;
		}
	}

	if($vflg) {print "num1:\t$_\n";}

	s/^\s*//;
	if(/\d/ && !/^<\/?[spa]/)		# opt and protect sgml
	{	@input = split(/\s+/o);
		@output=();
	for($field=0;$field<=$#input;$field++)	# $field is global
		{	if($field>0) {$last=$input[$field-1];}
			else {$last='';}
			if($field<$#input) {$next=$input[$field+1];}
			else {$next='';}
			if($field<$#input-1) {$next2=$input[$field+2];}
			else {$next2='';}
			$this=$input[$field];
			$_=$input[$field];
	
			if(/<[\w\.\/]*>/o && !/<p/o && !/<\/p>/o) # pass only
				{&perr("spurious SGML: $_"); next; }	# <p... and </p>
	
			if(/[0-9]/o && !/<p/o)		# number but not <p
			{	if(/[\$\#]/o)			# money
					{if (! &money($_,$next)) {next;} }
				elsif(/\d:\d\d$/o || /\d:\d\d\D/o)	# time
					{if (! &printtime($_)) {next;} }
				elsif(/\d+\/\d+\/\d+/o)		# x/x/x date
					{if (! &printdate($_)) {next;} }
				elsif((/[a-zA-Z].*\d/ || /\d.*[a-zA-Z]/)
				      && 
				      !(/\dth\W*/i || /1st\W*/i || /2nd\W*/i
					|| /3rd\W*/i
					|| (/\d\'?s\W*/
					    && (! /\d[a-zA-Z]+\d+\'?s\W*$/))))
					{if (! &printserno($_)) {next;} }	 # serial no
				elsif(/\//o)			# fraction
					{if (! &printfrac($_)) {next;} }
				elsif(/\d\'-?\d+/o)		# ft inches
					{if (! &printftin($_)) {next;} }
				else {if (! &printnum($_)) {next;} }	      # ordinary number
			}
			else {&pusho($_ );}		# non-numeric string
		}
		$_=join(" ",@output);
	}
	s/^/ /o;
	s/$/ /o;
	s/ - /-/g;		# unspace hyphen
	s/%/ % /g;
	s/ {2,}/ /g;
	s/^ //o;
	s/ $//o;

	if($_) {print "$_\n";}
}

sub money				# money($this,$next)
{	$_=$_[0];		# $this
	local($next)=$_[1];
	if($vflg) {print "money: $_, $next\n";}

	local($unit);
	local($subunit_sing);
	local($subunit_pl);
	local($punct);
	local($plural);
	local($sing);
	local($frac);
	local($front);
	local($back);
	local($x);
	local($y);
	local($z);
	local($i);
	local($j);

	s/\$\.(\d)/\$0.$1/g;	# patch numbers like $.22
	if(/A\$/)				# $ stuff
	{	($front)=/^(.*)A\$/;
		s/A\$//;
		$unit='Australian dollar';
		$subunit_sing='cent';
		$subunit_pl='cents';
	}
	elsif(/C\$/)
	{	($front)=/^(.*)C\$/;
		s/C\$//;
		$unit='Canadian dollar';
		$subunit_sing='cent';
		$subunit_pl='cents';
	}
	elsif(/NZ\$/)
	{	($front)=/^(.*)NZ\$/;
		s/NZ\$//;
		$unit='New Zealand dollar';
		$subunit_sing='cent';
		$subunit_pl='cents';
	}
	elsif(/US\$/)
	{	($front)=/^(.*)US\$/;
		s/US\$//;
		$unit='U S dollar';
		$subunit_sing='cent';
		$subunit_pl='cents';
	}
	elsif(/\$/)
	{	($front)=/^(.*)\$/;
		s/\$//;
		$unit='dollar';
		$subunit_sing='cent';
		$subunit_pl='cents';
	}
	elsif(/#/)				# pound
	{	($front)=/^(.*)#/;
		s/#//;
		$unit='pound';
		$subunit_sing='penny';
		$subunit_pl='pence';
	}
	else {&perr("money: unknown currency"); return 0;}

	($back)=/(\D*)$/;
	$back =~ s/^s//;	# $40s -> $40

	if($front) 
	{	&pusho($front);			# generally punctuation
		if($front !~ /\w$/) {$appendflg=1;}
	}

	$x=$_;
	if($x =~ /\//)
	{	$x =~ s/^\D*//;
		$x =~ s/\D*$//;
		if (! &printfrac($x)) {return 0;}
		&pusho("of a $unit");
		$x="";
		$plural=0;
	}

	$x =~ s/^\D*([\d,]*)\D*.*$/$1/;		# int part of string
	if($x ne "") {if (! &printint($x)) {return 0;} }		# print int part (eg. dollars)

	if($next eq "and" && $next2 =~ /\d\/\d/ && next2 !~ /\/.*\//)
	{	if($unit && $x ne "") {&pusho("and");}	      # frac: eg 4 1/16
		$z=$next2;
		$z =~ s/\D*$//;
		if (! &printfrac($z)) {return 0;}
		($punct)=($next2 =~ /(\D*)$/);
		$field+=2;
		&pusho("${unit}s");
	
		if($back) {&perr("money: back and 1 1/3"); return 0;}
		
		if($punct) {&appendo($punct);}	# punctuation from *illion
		return 1;
	}

	if($back eq "" && $next =~ /^(thousands?|[a-z]*illions?)(\W*)/i)
	{	if (! &printdecfrac($_)) {return 0;}			# multiplier
		&pusho($1);
		$punct=$2;
		$plural=1;			### if adj '', if noun 's'
		$field++;
		$frac=1;
	}
	elsif(/\.\d$/ || /\.\d\D/ || /\.\d{3}/ )	# .d or .ddd+
	{	if (! &printdecfrac($_)) {return 0;}
		$plural=1;			# can be either
		$frac=1;
	}
	else
	{	$y=$x;
		$y =~ s/,//g;			# remove commas
		if(int($y)!=1) {$plural=1;}
	}

	if($back eq "" && $input[$field+1] =~ /dollar/i)
	{	$unit="";			# fix "$1 dollar" wsj typo
		$subunit_sing="";
		$subunit_pl="";
		if (! &printdecfrac($_)) {return 0;}
		$frac=1;
	}

#print "f=$front, m=$_, b=$back\n";
#foo
	$sing=0;
	if($last =~ /^\W*[aA][nN]?\W*$/) {$sing=1;}	# a $123, an $80
	elsif($input[$field+1] eq "-") {$sing=1;}	# eg. $123-a-day
							# next one is chancy
	elsif($input[$field] !~ /\W$/ && $input[$field+1] !~ /^\W/ &&
		$input[$field+1] =~ /[a-zA-Z]$/ && $input[$field+2] eq "-" &&
		$input[$field+3] =~ /^[a-zA-Z]/) {$sing=1;}	# $ after-tax

	elsif($back eq "" && !$punct) # right contexts with no intervening punct
	{	$j=$field+1;		# includes *ly as a skip
		$z="";
		for($i=0;$i<2;$i++,$j++)	# skip ?
		{	$y=$input[$j];			# strip final punct
			$y =~ s/\W*$//;
			if($y !~ /\w*ly$/i && $sing_dollar{$y}!=2) {last;}
			($y)=($input[$j] =~ /(\W*)$/);	# get final punct
			$z .= $y;			# accumulate
		}
		$y=$input[$j];			# strip final punct
		$y =~ s/\W*$//;
		if($z eq "" && $sing_dollar{$y}==1) {$sing=1;}
	}
		
	if($unit)					# print unit
	{	&pusho($unit);
		if($plural && !$sing) {&appendo("s");}	# just add s for plural
	}

	if(!$frac && /\.\d{2}/)			# .dd	(eg. cents)
	{	$y=$_;
		$y =~ s/^[^\.]*\.([\d]*)\D?.*$/$1/;	# get fractional part
		if($unit && $x ne "") {&pusho("and");}
		if (! &printint($y)) {return 0;}
		if($sing || int($y)==1) {&pusho($subunit_sing);}
		else {&pusho($subunit_pl);}
	}

	if($back)				# punctuation from this field
	{	if($punct) {&perr("money: back and punct"); return 0;}

		if($back =~ /^\w/) {&pusho($back);}
		else {&appendo($back);}
	}
		
	if($punct) {&appendo($punct);}		# punctuation from *illion

  return 1;
}

sub printyear			# &printyear(x)
{	if($vflg) {print "printyear: $_[0]\n";}
	return &printnum($_[0]);		# for now
}

sub printtime			# &printtime(x)
{	if($vflg) {print "printtime: $_[0]\n";}
	$_=$_[0];
	
	local(@x);
	local($front);
	local($back);

	if(/:{2,}/ || !/\d:\d/) {&perr("printtime: not a time"); return 0;}

	@x=split(/:/,$_);
	($front)=($x[0] =~ /^(\D*)/);
	$x[0] =~ s/^(\D*)//;
	($back)=($x[1] =~ /(\D*)$/);
	$x[1] =~ s/(\D*)$//;
	
	if($front) 
	{	&pusho($front);			# generally punctuation
		if($front !~ /\w$/) {$appendflg=1;}
	}
	if (! &printint($x[0])) {return 0;}
	if($x[1]==0)
	{	$_=$next;
		if(!/^[aApP]\.?[nM]\.?$/) {&pusho("o'clock");}
	}
	elsif ($x[1]<10)
	{	&pusho("oh");
		if (!&printint($x[1])) {return 0;}
	}
	else {if (! &printint($x[1])) {return 0;} }
	if($back)
	{	if($back =~ /^\w/) {&pusho($back);}
		else {&appendo($back);}		# generally punctuation
	}
  return 1;
}

sub printfrac
{	if($vflg) {print "printfrac: $_[0]\n";}
	local($x)=$_[0];

	local(@z);			#Perl BUG: lists do not seem to be local
	local($sign);
	local($front);
	local($back);
	local($sign);

	$x =~ s/^([^\d\.]*)//;		# strip front
	$front=$1;
	if($front =~ /^\+$/)		# get sign
	{	$sign="plus";
		$front =~ s/\+$//;
	}
	if($front =~ /^-$/)
	{	$sign="minus";
		$front =~ s/-$//;
	}

	if($x =~ /\D$/)
	{	($back)=( $x =~ /(\D*)$/ );
		$x =~ s/\D*$//;			# strip back: final . is punct
	}

	@z=split(/\//,$x);
	if($#z !=1) {&perr("printfrac: illegal fraction: $_[0]"); return 0;}
	if($z[1] <= 1) {&perr("printfrac: den too small: $_[0]"); return 0;}

	if($front) 
	{	&pusho($front);
		if($front =~ /[a-zA-Z]$/) {&appendo("-");}
		$appendflg=1;
	}

	if($sign) {&pusho($sign);}

	if (! &printint($z[0])) { return 0;}			#numerator
	if($z[1] <= $#den)			# small den from table (<20)
	{	&pusho($den[$z[1]]);
		if($z[0]!=1) {if (! &pluralize) {return 0;} }
	}
	else					#large den
	{	$ones=int($z[1]%100);
		$hun=100*int($z[1]/100);
		if($hun>0) {if (!&printint($hun)) {return 0;} }
		if($ones==0) 
		{	&appendo("th");
			if($z[0]!=1) {if (! &pluralize) {return 0;} }
		}
		elsif($ones<=$#largeden)		# <20
		{	&pusho($largeden[$ones]);
			if($z[0]!=1) {if (!&pluralize) {return 0;} }
		}
		else
		{	$x=int($ones%10);
			if(int($ones/10))
			{	&pusho($ten[int($ones/10)]);
				if($x)
				{	&appendo("-");	# eg. twenty-five
					$appendflg=1;
				}
			}
			if($x==0)
			{	&pusho("th");
        if($z[0]!=1) {if (! &pluralize) {return 0;} }
			}
			else
			{	&pusho($largeden[$x]);
        if($z[0]!=1) {if (! &pluralize) {return 0;} }
			}
		}
	}

	if($back)
	{	$x=&geto;	# in case of 1/10th etc ([stndrth]=st nd rd th)
		if($back !~ /^[stndrth]{2}/ || $x !~ /$back$/)
		{	if($back =~ /^[a-zA-Z]/) {&appendo("-");}
			&appendo($back);
		}
	}
  
  return 1;
}

sub printnum			# printnum(n)
{	if($vflg) {print "printnum: $_[0]\n";}
	local($x)=$_[0];	# print ordinary numbers

	$leadingzeroflg='';			# global
	local($front);
	local($back);
	local($intpart);
	local($fracpart);
	local($hun);
	local($ones);
	local($comma);
	local($sign);
	local($y);

	$x =~ s/^(\D*)//;		# strip front
	$front=$1;
	if($front =~ /^\.$/ || $front =~ /\W\.$/ ||
		($front =~ /\.$/ && $x =~ /^0/ ))		# leading .
	{	$front =~ s/\.$//;
		$x = "." . $x;
	}
	if($front =~ /^\+$/)		# get sign
	{	$sign="plus";
		$front =~ s/\+$//;
	}
	if($front =~ /^-$/)
	{	$sign="minus";
		$front =~ s/-$//;
	}

	if($x =~ /\D$/)
	{	$back=$x;
		$back =~ s/^[\d\.,]*\d//;
		$x =~ s/\D*$//;			# strip back: final . is punct
	}

	if($x =~ /[^\d\.,]/) {&perr("printnum: $_[0] is not a number"); return 0;}

	if($x!=0 && $x =~ /^0/ && $x =~ /^\d*$/)	# "oh" numbers
	{	if($front) 
		{	&pusho($front);
			if($front !~ /[a-zA-Z]$/) {$appendflg=1;}
		}

		if($sign) { &pusho($sign); }
	
		while($x ne '')
		{	$x =~ s/^(.)//;
			&pusho($ones_oh[$1]);
		}

		if($back)
		{	if($back =~ /^s$/ || $back =~ /^s\W/)	# back = s
			{	if (! &pluralize) {return 0;}			# eg. 1960s
				$back =~ s/^s//;
			}
			if($back)
			{	if($back =~ /^[a-zA-Z]/) {&pusho($back);}
				else {&appendo($back);}	# back = punct or "'s"
			}
		}
		return 1;
	}

	if($x =~ /^\d/)			# get integer part
	{	if($x =~ /,/)
		{	$comma=1;
			$x =~ s/,//g;	# strip commas
		}
		$intpart=$x;
		$intpart =~ s/\..*$//;
		if($x =~ /^0/) {$leadingzeroflg=1;}
	}

	if($x =~ /\./)			# get fractional part
	{	$fracpart=$x;
		$fracpart =~ s/^.*\././;
	}

	if($front) 
	{	&pusho($front);
		if($front !~ /[a-zA-Z]$/) {$appendflg=1;}
	}

	if($sign) { &pusho($sign); }

	$ones=int($intpart%100);
	if($comma) {if (! &printint($intpart)) {return 0;} }
	elsif(($intpart>=1900 || $intpart>=1100 && $ones==0)
		&& $intpart<2000 && !$fracpart)			#4 digit -> 2+2
	{	$hun=int($intpart/100);
		if (! &printint($hun)) {return 0;}
		if($ones>=10) {if (! &printint($ones)) {return 0;} }
		elsif($ones>0)
		{	&pusho("oh");
			if (! &printint($ones)) {return 0;}
		}
		else {&pusho("hundred");}
	}
	else
	{	if (! &printint($intpart)) {return 0;}
		$y=$last;
		$y =~ s/^\W*//;				# thize dates: May 25th
		if(length($intpart)<=2 && $months{$y})
		{	if (! &thize("")) {return 0;}
			$back =~ s/[a-z]//g;
		}
	}
	if($fracpart) {if (! &printdecfrac($fracpart)) {return 0;} }

	if($back)
	{	if($back =~ /^s$/ || $back =~ /^s\W/)	# back = s
		{	if (! &pluralize) {return 0;}			# eg. 1960s
			$back =~ s/^s//;
		}
		if($back =~ /^st$/ || $back =~ /^st\W/)	# back= st
		{	if (! &thize("st")) {return 0;}			# eg. 1st
			$back =~ s/^st//;
		}
		if($back =~ /^nd$/ || $back =~ /^nd\W/)	# back= nd
		{	if (! &thize("nd")) {return 0;}			# eg. 2nd
			$back =~ s/^nd//;
		}
		if($back =~ /^rd$/ || $back =~ /^rd\W/)	# back= rd
		{	if (! &thize("rd")) {return 0;}			# eg. 3rd
			$back =~ s/^rd//;
		}
		if($back =~ /^th$/ || $back =~ /^th\W/)	# back= th
		{	if (! &thize("th")) {return 0;}			# eg. 4th
			$back =~ s/^th//;
		}
		if($back)
		{	if($back =~ /^[a-zA-Z]/) {&pusho($back);}
			else {&appendo($back);}	# back = punct or "'s"
		}
	}
  return 1;
}

sub printdate			# printdate(n):	x/x/x format
{	if($vflg) {print "printdate: $_[0]\n";}
	local($x)=$_[0];	# print ordinary numbers

	local(@y);
	local($front);
	local($back);

	$x =~ s/^(\D*)//;		# strip front
	$front=$1;

	$x =~ s/(\D*)$//;		# strip back
	$back=$1;

	if($x !~ /^\d{1,2}\/\d{1,2}\/(19)?\d{2}$/)
		{&perr("printdate: $_[0] is not a date"); return 0;}

	@y=split(/\//,$x);
	$y[2] =~ s/^19(\d{2})$/$1/;
	
	if($front) 
	{	&pusho($front);
		if($front =~ /[a-zA-Z]$/) {&appendo("-");}
		$appendflg=1;
	}

	if (! &printint($y[0])) {return 0;}
	&appendo("/");

	$appendflg=1;
	if (! &printint($y[1])) {return 0;}
	&appendo("/");

	$appendflg=1;
	if (! &printint($y[2])) {return 0;}

	if($back)
	{	if($back =~ /^[a-zA-Z]/) {&appendo("-");}
		&appendo($back);
	}
  return 1;
}

sub printserno			# printserno(n): eg. B1, 3b2, 10W-40
{	if($vflg) {print "printserno: $_[0]\n";}
	local($x)=$_[0];	# print mixed sequences of dig and let

	local($y);
	local($z);
	local($front);
	local($back);

	$x =~ s/^(\W*)//;		# strip front
	$front=$1;
	if($front) 
	{	&pusho($front);
		if($front !~ /[a-zA-Z]$/) {$appendflg=1;}
	}

	$x =~ s/(\W*)$//;		# strip back
	$back=$1;
	$x =~ s/(\d[a-zA-Z]+\d+)(\'?s)$/$1/  # strip "s" or "'s"
	    && ($back = $2 . $back);

	while($x)
	{	$x =~ s/^(\D*)//;	# strip off non-dig
		$y=$1;
		if($y)
		{	$y =~ s/-//g;	# remove -
			if($y eq "") {}
			elsif($sernowd{$y}) {&pusho($y);}	# word
			else
			{	while($y)			# spell out
				{	if($y =~ /[a-zA-Z]\'s$/)
					{	&pusho($y);
						$y =~ s/[a-zA-Z]\'s*$//;
					}
					elsif($y =~ /[A-Z]s$/)
					{	&pusho($y);
						$y =~ s/[A-Z]s$//;
					}
					else
					{	$y =~ s/^(.\.?)//;
						&pusho($1);
					}
				}
			}
		}		     # (should expand here unless in dictionary)
		$x =~ s/^(\d*)//;	# strip off dig
		$y=$1;
		if($y ne "") { if (! &printdigstr($y)) {return 0;} }
	}

	if($back =~ /^s\b/)	# back = s
	{			# eg. 2C60s
	    if (! &pluralize) {return 0;} 
	    $back =~ s/^s//;
	}
	if($back)
	{	if($back =~ /^\w/) {&pusho($back);}
		else {&appendo($back);}
	}
	$appendflg=0;
  return 1;
}

sub printdigstr			# printdigstr(x)
{	if($vflg) {print "printdigstr: $_[0]\n";}
	local($x)=$_[0];

	local(@y);
	local($j);
	local($k);

	if($x =~ /^0/)			# leading zero
	{	while($x ne "")
		{	$x =~ s/^(.)//;
			if($1 !~ /\d/) {&perr("printdigstr: non-digit"); return 0;}
			&pusho("$ones_z[$1]");
		}
		return;
	}
	if($x =~ /^\d0*$/)		# d, d0, d00, d000, etc
	{	return &printint($x);
	}

	$_=$x;
	@y=();
	for($j=0;$_ ne "";$j++) { $y[$j]=chop($_); }	# j=no digits
	for($k=0;$y[$k]==0;$k++) {}			# k= nr following 0s

	if($j==2)			# 2 dig
	{	return &printint($x);
	}
	if($j==3)
	{	if (! &printint($y[2])) {return 0;}
		if($y[1]==0) {&pusho("oh");}
		return &printint("$y[1]$y[0]");
	}
	if($j==5 && $k<=2)
	{	if (! &printint("$y[4]")) {return 0;}
		$j=4;
	}
	if($j==4)
	{	if (! &printint("$y[3]$y[2]")) {return 0;}
		if($k==2) {&pusho("hundred");}
		else
		{	if($y[1]==0) {&pusho("oh");}
			return &printint("$y[1]$y[0]");
		}
		return 1;
	}
						# >5 dig: just sequential dig
	for($j--;$j>=0;$j--) {&pusho("$ones_oh[$y[$j]]");}
  return 1;
}

sub printftin			# printftin(n): eg. 6\'-4\"
{	if($vflg) {print "printftin: $_[0]\n";}
	local($x)=$_[0];	# print mixed sequences of dig and let

	local($y);
	local($front);
	local($back);

	$x =~ s/^(\D*)//;		# strip front
	$front=$1;

	$x =~ s/(\D*)$//;		# strip back
	$back=$1;
	$back =~ s/^\"//;		# remove \"

	if($front) 
	{	&pusho($front);
		if($front !~ /[a-zA-Z]$/) {$appendflg=1;}
	}

	$x =~ s/^([\d\.]*)//;	# strip off dig & .
	$y=$1;
	if(!$y) {&perr("printftin: bad feet"); return 0;}
	if (! &printnum($y)) {return 0;}
	if($y==1) {&appendo("-foot");}
	else {&appendo("-feet");}

	$x =~ s/^\'//;	# strip off \'
	$x =~ s/^-//;	# strip off -
	if(!$x) {&perr("printftin: bad intermed"); return 0;}

	$x =~ s/^([\d\.]*)//;	# strip off dig & .
	$y=$1;
	if(!$y) {&perr("printftin: bad inches"); return 0;}
	if (! &printnum($y)) {return 0;}
	if($y==1) {&appendo("-inch");}
	else {&appendo("-inches");}

	if($back)
	{	if($back !~ /^[a-zA-Z]/) {&appendo($back);}
		else {&pusho($back);}
	}
  return 1;
}

sub printint			# printint(x)
{	if($vflg) {print "printint: $_[0]\n";}
	local($x)=$_[0];

	local($comma);
	local($leading_zero);
	local($fractional);
	local(@y);
	
	$fractional=$x =~ /\.\d/;
	$x =~ s/^\D*([\d,]*)\D*.*$/$1/;	# int part of string
	$leading_zero=$x =~ /^0/;
	$comma=$x =~ /,/;
	$x =~ s/,//g;
	if($x eq "") {return;}

	if($x == 0)
	{	&pusho("zero");
		$leadingzeroflg=1;
		return;
	}
	
	@y=();
	for($j=0;$x;$j++) { $y[$j]=chop($x); }

	if($comma || $fractional || 1)
	{	for($j=3*int($#y/3);$j>=0;$j-=3)
		{	if($y[$j+2]) { &pusho("$ones_z[$y[$j+2]] hundred");}
			if($y[$j+1]==1) { &pusho($teen[$y[$j]]);}
			else
			{	if($y[$j+1]>1)
				{	&pusho($ten[$y[$j+1]]);
					if($y[$j])
					{	&appendo("-");	# twenty-five
						$appendflg=1;
					}
				}
				if($y[$j]>0) { &pusho($ones_z[$y[$j]]);}
			}
			if(int($j/3)>0)
			{	if(int($j/3) > $#mult)
					{ &perr("printint: too big"); return 0;}
				&pusho($mult[int($j/3)]);
			}
			$commanextflg=1;
		}
	}
	$commanextflg=0;
  return 1;
}

sub printdecfrac
{	if($vflg) {print "printdecfrac: $_[0]\n";}
	local($x)=@_[0];
	
	if($x !~ /\.\d/) {return;}
	$x =~ s/^[^\.]*\.([\d]*)\D?.*$/$1/;		# get fractional part

	&pusho($POINT);
	@y=split(//,$x);
	if($leadingzeroflg)
		{for($j=0;$j<=$#y;$j++) { &pusho($ones_z[$y[$j]]);}}
	else {for($j=0;$j<=$#y;$j++) { &pusho($ones_oh[$y[$j]]);}}

  return 1;
}

sub pluralize		# pluralize(): pluralize last entry on output stack
{	if($vflg) {print "pluralize: $_[0]\n";}
	local($x);

	$_=&geto;
	if( /st$/ || /nd$/ || /rd$/ || /th$/ || /quarter$/ || /zero$/ || /oh/ ||
		/one$/ || /two$/ || /three$/ || /four$/ || /five$/ ||
		/seven$/ || /eight$/ || /nine$/ ||
		/ten$/ || /eleven$/ || /twelve$/ || /een$/ ||
		/hundred$/ || /thousand$/ || /illion$/ )
	{	&appendo("s");
	}
	elsif (/six$/)
	{	&appendo("es");
	}
	elsif (/half$/)
	{	$x=&popo();
		$x =~ s/f$/ves/;
		&pusho($x);
	}
	elsif (/ty$/)			# fifty etc.
	{	$x=&popo();
		$x =~ s/y$/ies/;
		&pusho($x);
	}
	else {&perr("pluralize: unknown word: $_"); return 0;}

  return 1;
}

sub thize		# thize(): add th to last entry on output stack
{	if($vflg) {print "printthize: $_[0]\n";}
	local($y)=$_[0];

	local($x);

	$_=&geto;
	if( /four$/ || /six$/ || /seven$/ || /ten$/ ||
		/eleven$/ || /een$/ || /hundred$/ || /thousand$/ || /illion$/ )
	{	if($y && $y ne "th") {&perr("thize: mismatch: $_ $y\n"); return 0;} # xth
		&appendo("th");
	}
	elsif( /one$/ )						# 1st
	{	if($y && $y ne "st") {&perr("thize: mismatch: $_ $y\n"); return 0;}
		$x=&popo();
		$x =~ s/one$/first/;
		&pusho($x);
	}
	elsif( /two$/ )						# 2nd
	{	if($y && $y ne "nd") {&perr("thize: mismatch: $_ $y\n"); return 0;}
		$x=&popo();
		$x =~ s/two$/second/;
		&pusho($x);
	}
	elsif( /three$/ )					# 3rd
	{	if($y && $y ne "rd") {&perr("thize: mismatch: $_ $y\n"); return 0;}
		$x=&popo();
		$x =~ s/three$/third/;
		&pusho($x);
	}
	elsif( /five$/ || /twelve$/ )				# 5th, 12th
	{	if($y && $y ne "th") {&perr("thize: mismatch: $_ $y\n"); return 0;}
		$x=&popo();
		$x =~ s/ve$/fth/;
		&pusho($x);
	}
	elsif(/eight$/)
	{	if($y && $y ne "th") {&perr("thize: mismatch: $_ $y\n"); return 0;} # 8th
		&appendo("h");
	}
	elsif( /nine$/ )
	{	if($y && $y ne "th") {&perr("thize: mismatch: $_ $y\n"); return 0;}
		$x=&popo();
		$x =~ s/nine$/ninth/;
		&pusho($x);
	}
	elsif( /ty$/ )
	{	if($y && $y ne "th") {&perr("thize: mismatch: $_ $y\n"); return 0;}
		$x=&popo();
		$x =~ s/ty$/tieth/;
		&pusho($x);
	}
	else {&perr("thize: unknown word: $_"); return 0;j}
  return 1;
}

sub pusho				# pusho($x): push output
{	if($commanextflg)		# global: used for commas in printint
	{	$commanextflg=0;		
		&appendo(",");
	}
	if($appendflg)			# global: used for fronts
	{	$appendflg=0;		
		&appendo(@_[0]);
	}
	else {push(@output,@_);}
}

sub appendo				# appendo($x): append to output
{	$appendflg=0;		
#	if($#output < 0) {&pusho("");}
	if($#output < 0) {&perr("appendo: output empty"); return 0;}
	$output[$#output] .= @_[0];
}

sub popo				# popo(): pop last output
{	if($#output < 0) {&perr("popo: output empty"); return 0;}
	pop(@output);
}

sub geto				# geto(): get last output
{	if($#output < 0) {&perr("geto: output empty"); return 0;}
	return $output[$#output];
}

sub perr
{	print STDERR "numproc: $_[0]\n";
	print STDERR "line number=$.: fields=$last, $this, $next\n";
#	exit(1);

	$appendflg=0;
	$commanextflg=0;
	&pusho($this);
}

sub perr2
{	print STDERR "numproc: $_[0]\n";
	exit(1);
}
