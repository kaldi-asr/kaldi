#!/usr/bin/env bash

# This script replaces the command readlink -f (which is not portable).
# It turns a pathname into an absolute pathname, including following soft links.
target_file=$1

cd $(dirname $target_file)
target_file=$(basename $target_file)

# Iterate down a (possible) chain of symlinks
while [ -L "$target_file" ]; do
    target_file=$(readlink $target_file)
    cd $(dirname $target_file)
    target_file=$(basename $target_file)
done

# Compute the canonicalized name by finding the physical path 
# for the directory we're in and appending the target file.
phys_dir=$(pwd -P)
result=$phys_dir/$target_file
echo $result
