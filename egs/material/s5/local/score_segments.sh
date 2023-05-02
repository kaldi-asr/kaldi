#!/usr/bin/env bash
set -e -o pipefail
set -o nounset                              # Treat unset variables as an error

echo "$0" "$@"
local/score_wer_segments.sh "$@"
#local/score_cer_segment.sh --stage 2 "$@"

