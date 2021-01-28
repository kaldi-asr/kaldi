#!/bin/bash -u

. ./cmd.sh
. ./path.sh

ICSI_TRANS=/export/corpora3/LDC/LDC2004T04/icsi_mr_transcr #where to find ICSI transcriptions [required]
. utils/parse_options.sh
set -euxo pipefail

#prepare dictionary and language resources
local/ICSI/icsi_prepare_dict.sh

#prepare annotations, note: dict is assumed to exist when this is called
local/ICSI/icsi_text_prep.sh $ICSI_TRANS data/local/ICSI_annotations
echo "Done"
exit 0

