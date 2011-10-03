#!/usr/bin/perl

# This script is like "echo", except that if a command-line argument
# is empty or has a space in, it will enclose it in quotes.  It uses
# double quotes by default, or double quotes if the string contains
# both spaces and single quotes.

foreach $x (@ARGV) { 
    if ($x =~ m/^\S+$/) {  print $x . " "; }
    elsif ($x =~ m:\":) { print "'\''$x'\'' "; }
    else { print "\"$x\" "; } 
}
print "\n";

