#!/bin/bash

# Copyright 2012  Daniel Povey.  Apache 2.0.
# Creates the file tools.dox from tools.dox.input
# and the source code.
# to be run from ..

echo "Making tools.dox from source files"
[ ! -f doc/make_tools.sh ] && echo "You are running this script from the wrong directory" && exit 1;


for d in *bin; do
  if [ -d $d ] && [ -f $d/Makefile ]; then
    cat $d/Makefile | perl -ane ' while(<>){ s/\\\n//g; print; }' | grep -E '^BINFILES' | awk '{for(n=3;n<=NF;n++){print $n;}}' > tmpf;
    for binfile in `cat tmpf`; do 
        perl -e 'use File::Basename;  $/ = "zyx"; $f=$ARGV[0]; while(<>) {
      m/char\s*\*\s*usage\s*=((\s*\".+\"\s*\n)*(\s*\".+\";\s*\n))/ || die "bad $f\n"; #=\s*(\"[.\n]+\");\s*\n/ || die "could not find usage message for file $f\n"; 
      $msg = $1;  
      $msg =~ s/^\s*\"//g || die "(file is $f)"; # Remove initial quotes.
      $msg =~ s/\";\s*$//g || die "(file is $f)"; # Remove final quotes.
      $msg =~ s/\"\s*\n\s*\"//g; # remove intermediate quotes.
      $msg =~ s/\\\"/\"/g; # Un-escape escaped quotes.
      $msg =~ s/\\n/\n/g; # Turn escaped newlines into newlines.
      $msg =~ s/\\\t/\t/g; # Turn escaped tabs into tabs.
      $msg =~ s/\n\s*$//g; # Remove the final newline.
      $basef = basename($f);
      $output = "<tr> <td> \\ref $f \"$basef\" </td><td> $msg </td> </tr>";
      $output =~ s|\n|<br />|g; # make it so newlines are marked up.
      print "$output\n";
     } ' $d/$binfile.cc
    done
  fi
done > doc/table;

! perl -e '$/ = "xyicfab"; open(F, "<$ARGV[0]") || die "opening $ARGV[0]";
   open(G, "<$ARGV[1]") || die "opening $ARGV[1]";
   $table = <G>;
   $file = <F>;
   $file =~ s:PUT_DATA_HERE:$table: || die "No match!";
   print $file; ' doc/tools.dox.input doc/table >doc/tools.dox \
  && echo "Making tools.dox did not work." && exit 1;




