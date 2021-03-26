#!/usr/bin/env bash

echo "$0: finding missing inter-directory dependencies in src/Makefile"

cd src

for x in */Makefile; do
  dir=$(dirname $x); 
  for dependency in $(perl -ape 's/\\\n//;' <$x | grep ADDLIBS | awk '{$1="";$2="";print;}' | perl -ane 'print "$1\n" while ( $_ =~ m|\.\./([^/]+)/|g ); '); do
    if ! perl -ape 's/\\\n//;' <Makefile | grep -E "\b$dir\b.*:" | grep -w $dependency >/dev/null; then
      echo "$dir: $dependency"; 
    fi
  done
done
