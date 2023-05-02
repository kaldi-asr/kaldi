#!/usr/bin/env bash

# This script is called upon a test failure under Travis CI to report the
# failure context. It prints a header followed by the last 200 lines of every
# *.testlog file found recursively under ./src. src/makefiles/default_rules.mk,
# in turn, keeps *.testlog files from failed tests only.

for f in $(find src/ -name '*.testlog' | sort); do
  printf "
********************************************************************************
* %-76s *
********************************************************************************
" $f
  tail --lines=200 $f
done
