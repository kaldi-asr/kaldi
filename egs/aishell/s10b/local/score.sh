#!/bin/bash

set -e -o pipefail
set -x
steps/score_kaldi.sh "$@"
steps/scoring/score_kaldi_cer.sh --stage 2 "$@"

echo "$0: Done"
