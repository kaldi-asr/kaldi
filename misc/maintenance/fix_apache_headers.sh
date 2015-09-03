#!/bin/bash

# makes sure the line See ../../COPYING for clarification regarding multiple
# authors appears in the apache headers in the source, and that source files
# have their Apache headers.  Including this mainly for documentation, as I
# doubt the issue will occur much in future.

# run this from the top level of the repo, as
# misc/maintenance/fix_apache_headers.sh

set -e
cd src
rm -rf tmp
for x in */*.{h,cc,dox}; do 
  if [ $x != "util/basic-filebuf.h" ]; then
    if ! grep 'COPYING for clarification' $x >/dev/null; then
      echo Fixing $x;
      if ! grep "Apache License" $x >/dev/null; then
        echo "$0: warning: file $x may not have an Apache license header"
      else
        cp $x tmp; cat tmp | perl -ape ' if (m/Licensed under the Apache License/) { 
        print "// See ../../COPYING for clarification regarding multiple authors\n"; 
        print "//\n";} ' > $x;
      fi
    fi
  fi
done
