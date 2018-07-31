#!/bin/bash

models=""
for x in $*; do   models="$models tdnn_${x}";   done

local/chain/compare_wer_general.sh $models
