#!/bin/sed -f
#
# $Header: /home/srilm/CVS/srilm/man/scripts/man2html.sed,v 1.6 2008/02/06 18:18:44 stolcke Exp $
#

s,\\-,-,g
s,\\&,,g
s,\\\\,/BS/,g

# replace < ... >
s,&,\&amp;,g
s,<,\&lt;,g
s,>,\&gt;,g

# font changes
s,\\fB\([^\\]*\)\\fP,<B>\1</B>,g
s,\\fI\([^\\]*\)\\fP,<I>\1</I>,g

s,/BS/,\\,g

# crossrefs
s,\([A-Za-z][^ ]*\)(\([1-8]\)),<A HREF="\1.\2.html">&</A>,g
s,^\.BR  *\([A-Za-z][^ ]*\)  *(\([1-8]\)),<A HREF="\1.\2.html">\1(\2)</A>,g

