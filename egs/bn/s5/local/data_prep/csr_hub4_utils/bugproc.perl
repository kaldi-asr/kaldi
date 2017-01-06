#!/usr/bin/perl
# $Id: bugproc.perl,v 1.4 1996/08/21 23:55:40 robertm Rel $
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

# bugproc.comm
# Removes some bugs common to all sources.
# This script has no source-dependencies.

while(<>)
{
    if ( /^</ ) {  # pass all tag lines intact;
	print;
	next;
    }

    s/(\w)\(/$1 (/g;			# eg. x( -> x (
    s/\)(\w)/) $1/g;			# eg. )x -> ) x;

    s/(\d)\((\d)/$1 ($2/g;			# \d(\d
    s/(\d)\)(\d)/$1) $2/g;			# \d)\d;
    s/([a-zA-Z]{2,}\.)(\d)/$1 $2/g;		# eg. Sept.30
    s/,([a-zA-Z])/, $1/g;			# eg. 20,Smith
    s/(\W)milion(\W)/$1million$2/g;		# spelling err

    s/(\W&\s*)Co([^\w\.-])/$1Co.$2/g;		# "& Co" -> "& Co."
    s/(\WU\.S)([^\.\w])/$1.$2/g;		# U.S -> U.S.

    # next block added for Broadcast News archive processing
    s/\$ +(\d)/\$$1/g;		# e.g. "$ 5" -> "$5"
    s/\$\#/\$/g;		# e.g. "$#5" -> "$5" (typo??)
    s/\#/number /g;		# in bc-news, "#" = "number" not "pound"
    s=([^\s</])(/+)\s=$1 $2 =g;	# e.g. "2002/ " -> "2002 / "
    s=([0-9])/1,000([^0-9,])=$1/1000$2=g; # e.g. "1/1,000" -> "1/1000"

    s/\s{2,}/ /g;
    s/^ //;
    s/\s*$/ \n/;

    print;
}
