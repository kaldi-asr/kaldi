#!/bin/bash

# this script updates the .gitignore properties with the names of newly added binaries.

# First, here is some notes on how I updated .gitignore (at top level)
# using the previously listed things in the svn:ignore properties.

# # we'll first get a list of all directories in svn.
# svn list -R > listing
# grep '/$' listing > dirs

# for dir in $(cat dirs); do
#   for prop in $(svn propget svn:ignore $dir); do
#     echo $dir$prop
#   done
# done > bar

# # Then I edited the file after I noticed some things that shouldn't have been in .svignore.

# for x in $(cat bar); do if ! $(grep "^/$x$" .gitignore >/dev/null); then echo $x; fi; done

# # this is all I got.
# egs/callhome_egyptian/s5/mfcc
# egs/callhome_egyptian/s5/data
# egs/callhome_egyptian/s5/exp/egs
# egs/callhome_egyptian/s5/exp/src

# the rest of this file updates the .gitignore with the names of new binaries

svn list -R > listing

for f in $(grep '.cc$' listing); do
  binary=$(echo $f | sed s:.cc$::)
  if [ -f $binary ] && ! grep "^/$binary$" .gitignore >/dev/null; then
    echo /$binary 
  fi
done > new_binaries

cat new_binaries >> .gitignore

