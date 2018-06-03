#!/usr/local/bin/gawk -f
#
# $Header: /home/srilm/CVS/srilm/man/scripts/man2html.gawk,v 1.9 2007/12/20 20:29:10 stolcke Exp $
#

function getargs() {
	delete args;
	j = 1;
	for (i = 2; i <= NF; i ++) {
	    if ($i ~ /^"/) {
		args[j] = substr($i, 2);
		for (i++; i <= NF; i++) {
		    if ($i ~ /"$/) {
			args[j] = args[j] " " substr($i, 1, length($i)-1);
			break;
		    } else {
			args[j] = args[j] " " $i;
		    }
		}
	    } else {
		args[j] = $i;
	    }
	    j++;
	}
	$1 = "";
	allargs = "";
	if (j >= 1) {
	    for (k = 1; k <= j; k ++) {
		allargs = allargs " " args[k];
	    }
	}
	return j - 1;
}
function finishitem() {
	if (initem) {
		print "</DD>";
		initem = 0;
	}
}
function finishlist() {
	finishitem();
	if (inlist) {
		print "</DL>";
		inlist -= 1;
	}
}
function makelinks() {
	for (i = 1; i <= NF; i ++) {
		if ($i ~ /^http:/) {
			url = $i;
			sub("[,;.]$", "", url);
			$i = "<a href=\"" url "\">" $i "</a>";
		}
	}
}
function printline() {
	if (text ~ /\\c/) {
		sub("\\\\c", "", text);
		oldtext = oldtext text;
	} else {
		if (inlabel) {
			print "<DT>" oldtext text;
			print "<DD>";
			inlabel = 0;
			initem = 1;
			oldtext = "";
		} else {
			print oldtext text;
			oldtext = "";
		}
	}
}
# comments
$0 ~ /^\.\\\"/ {
	$1 = "";
	print "<!" $0 ">";
	next;
}
#
# Note: check two-letter macros before one-letter ones
#
# title
$0 ~ /^\.TH/ {
	getargs()
	name = args[1];
	print "<HTML>";
	print "<HEADER>";
	print "<TITLE>" name "</TITLE>";
	print "<BODY>"
	print "<H1>" name "</H1>";
	next;
}
# section header
$0 ~ /^\.SH/ {
	finishlist();
	getargs();
	print "<H2>" allargs "</H2>";
	next;
}
$0 ~ /^\.SS/ {
	finishlist();
	getargs();
	print "<H3>" allargs "</H3>";
	next;
}
$0 ~ /^\.PP/  || $0 ~ /^\.LP/ {
	finishlist();
	print "<P>";
	next;
}
$0 ~ /^\.br/ {
	print "<BR>";
	next;
}
# labeled list item
$0 ~ /^\.TP/ {
	if (!inlist) {
		print "<DL>";
		inlist += 1;
	}
	inlabel = 1;
	next;
}
# labeled list item
$0 ~ /^\.IP/ {
	getargs();
	print "<DT>";
	print allargs;
	print "<DD>";
	next;
}
# nested lists
$0 ~ /^\.RS/ {
	print "<DL>";
	inlist += 1;
	next;
}
$0 ~ /^\.RE/ {
	finishlist();
	next;
}
$0 ~ /^\.BR/ {
	getargs();
	text = "<B>" args[1] "</B>" args[2] "<B>" args[3] "</B>" args[4] "<B>" args[5] "</B>" args[6] "<B>" args[7] "</B>";
	printline();
	next;
}
$0 ~ /^\.BI/ {
	getargs();
	text = "<B>" args[1] "</B><I>" args[2] "</I><B>" args[3] "</B><I>" args[4] "</I><B>" args[5] "</B><I>" args[6] "</I><B>" args[7] "</B>";
	printline();
	next;
}
$0 ~ /^\.B/ {
	if (getargs() > 0) {
		text = "<B>" allargs "</B>";
	} else {
		text = "<B>";
		newfont="B";
	}
	printline();
	next;
}
$0 ~ /^\.IR/ {
	getargs();
	text = "<I>" args[1] "</I>" args[2] "<I>" args[3] "</I>" args[4] "<I>" args[5] "</I>" args[6] "<I>" args[7] "</I>";
	printline();
	next;
}
$0 ~ /^\.IB/ {
	getargs();
	text = "<I>" args[1] "</I><B>" args[2] "</B><I>" args[3] "</I><B>" args[4] "</B><I>" args[5] "</I><B>" args[6] "</B><I>" args[7] "</I>";
	printline();
	next;
}
$0 ~ /^\.I/ {
	if (getargs() > 0) {
		text = "<I>" allargs "</I>";
	} else {
		text = "<I>";
		newfont="I";
	}
	printline();
	next;
}
$0 ~ /^\.P/ {
	text = "</" newfont ">";
	printline();
	next;
}
$0 ~ /^\.RB/ {
	getargs();
	text = args[1] "<B>" args[2] "</B>" args[3] "<B>" args[4] "</B>" args[5] "<B>" args[6] "</B>" args[7];
	printline();
	next;
}
$0 ~ /^\.RI/ {
	getargs();
	text = args[1] "<I>" args[2] "</I>" args[3] "<I>" args[4] "</I>" args[5] "<I>" args[6] "</I>" args[7];
	printline();
	next;
}
$0 ~ /^\.nf/ {
	print "<PRE>";
	next;
}
$0 ~ /^\.fi/ {
	print "</PRE>";
	next;
}
$0 ~ /^\.[A-Za-z]/ {
	print "unknown macro " $1 > "/dev/stderr";
	next;
}
{
	makelinks();
	text = $0;
	printline();
	next;
}
END {
	print "</BODY>";
	print "</HTML>";
}

