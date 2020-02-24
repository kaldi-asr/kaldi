#!/usr/bin/env bash

# This script should be run from two levels up, as:
# misc/maintenance/svnignore.sh

# It takes the things listed in the .gitignore file (which is at
# the top level) and converts them into .svnignore properties
# in the subdirectories.

svn list -R > listing
grep '/$' listing > dirs
grep '^\*' .gitignore > patterns

for dir in $(cat dirs); do
  cp patterns cur_ignore
  grep -v '#' .gitignore | grep ^/$dir | sed s:^/$dir:: | sed s:/$:: >> cur_ignore
  echo .depend.mk >> cur_ignore
  svn propset -F cur_ignore svn:ignore $dir
done

