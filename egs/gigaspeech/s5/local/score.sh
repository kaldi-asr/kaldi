#!/usr/bin/env bash

set -e -o pipefail
set -x
steps/score_kaldi.sh "$@"

echo "$0: Done"
