#!/usr/bin/perl -pi.old-char

# handles nonprinting characters in Broadcast News material, to the extent
# that they can be handled, and perhaps a bit beyond...

tr/\xc4\x82\x90\xa4\x89\x8a\x87\xe9/-eEneece/;

s=\xae=<<=g;
s=\xaf=>>=g;
s=\xab= 1/2=g;
s=\xac= 1/4=g;
s=\xf8= degrees=g;
s=\xf1= plus or minus =g;
