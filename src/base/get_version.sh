#!/usr/bin/env bash

# Copyright 2017 University of Southern California (Author: Dogan Can)

# See ../../COPYING for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.


# Kaldi versioning is loosely based on the semantic versioning scheme. This
# script tries to work out the version number from the partial version number
# specified in src/.version along with the recent git history. By convention
# src/.version specifies the first two components (MAJOR.MINOR) of the version
# number. The third component (PATCH) is determined by counting how many
# commits there are that are newer than than the last commit modifiying
# src/.version. If there are uncommitted changes in the src/ directory, then
# the version number is extended with a suffix (~N) specifiying the number of
# files with uncommitted changes (including files that are not tracked by git).
# If git history is not available, then the version number defaults to the
# number specified in src/.version.

# Change working directory to the directory where this script is located.
cd "$(dirname ${BASH_SOURCE[0]})"

# Read the partial version number specified in the first line of src/.version.
version=$(head -1 ../.version)

if [[ $version != +([0-9]).+([0-9]) ]]; then
  echo "$0: The version number \"$version\" specified in src/.version is not" \
       "in MAJOR.MINOR format."
  echo "$0: Stopping the construction of full version number from git history."
elif ! which git >&/dev/null; then
  echo "$0: Git is not installed."
  echo "$0: Using the version number \"$version\" specified in src/.version."
elif [ "$(git rev-parse --is-inside-work-tree 2>/dev/null)" != true ]; then
  echo "$0: Git history is not available."
  echo "$0: Using the version number \"$version\" specified in src/.version."
else
  # Figure out patch number and head commit SHA-1.
  version_commit=$(git log -1 --pretty=oneline ../.version | cut -f 1 -d ' ')
  head_commit=$(git log -1 --pretty=oneline | cut -f 1 -d ' ')
  head_commit_short=$(git log -1 --oneline | cut -f 1 -d ' ')
  patch_number=$(git rev-list ${version_commit}..HEAD | wc -l)
  version="$version.$patch_number"

  # Check for uncommitted changes in src/.
  tracked_changes=$(git diff-index HEAD .. | wc -l)
  untracked_files=$(git ls-files --exclude-standard --others .. | wc -l)
  uncommitted_changes=$((tracked_changes + untracked_files))
  if [ $uncommitted_changes -gt 0 ]; then
    # Add suffix ~N if there are N files in src/ with uncommitted changes
    version="$version~$uncommitted_changes"
  fi
fi

# Empty version number is not allowed.
if [ -z "$version" ]; then
  version="?"
fi

# Write Kaldi version number info to ./version.h only if it is different from
# what is already there.
temp=$(mktemp)
trap 'rm -f "$temp"' EXIT
echo "// This file was automatically created by ./get_version.sh." > $temp
echo "// It is only included by ./kaldi-error.cc." >> $temp
echo "#define KALDI_VERSION_NUMBER \"$version\"" >> $temp
if [ -n "$head_commit" ]; then
  echo "#define KALDI_GIT_HEAD \"$head_commit\"" >> $temp
fi
if [ -n "${head_commit_short}" ]; then
  echo "#define KALDI_GIT_HEAD_SHORT \"${head_commit_short}\"" >> $temp
fi
if ! cmp -s $temp version.h; then
  cp $temp version.h
  chmod 644 version.h
fi
