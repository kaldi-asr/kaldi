#!/usr/bin/env bash

# This program searches within a directory for soft links that
# appear to be created by 'create_data_link.pl' to a 'storage/' subdirectory,
# and it removes both the soft links and the things they point to.
# for instance, if you have a soft link 
#   foo/egs/1.1.egs -> storage/2/1.1.egs
# it will remove both foo/egs/storage/2/1.1.egs, and foo/egs/1.1.egs.

ret=0

dry_run=false

if [ "$1" == "--dry-run" ]; then
  dry_run=true
  shift
fi

if [ $# == 0 ]; then
  echo "Usage:  $0 [--dry-run] <list-of-directories>"
  echo "e.g.: $0 exp/nnet4a/egs/"
  echo " Removes from any subdirectories of the command-line arguments, soft links that "
  echo " appear to have been created by utils/create_data_link.pl, as well as the things"
  echo " that those soft links point to.  Will typically be called on a directory prior"
  echo " to 'rm -r' on that directory, to ensure that data that was distributed on other"
  echo " volumes also gets deleted."
  echo " With --dry-run, just prints what it would do."
fi

for dir in $*; do
  if [ ! -d $dir ]; then
    echo "$0: not a directory: $dir"
    ret=1
  else
    for subdir in $(find $dir -type d); do
      if [ -d $subdir/storage ]; then
        for x in $(ls $subdir); do
          f=$subdir/$x
          if [ -L $f ] && [[ $(readlink $f) == storage/* ]]; then
            target=$subdir/$(readlink $f)
            if $dry_run; then
              echo rm $f $target
            else
              rm $f $target
            fi
          fi
        done
      fi
    done
  fi
done

exit $ret
