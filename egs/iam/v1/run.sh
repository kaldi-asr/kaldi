#!/bin/bash

stage=0

. ./cmd.sh
. ./path.sh

. utils/parse_options.sh

if [ $stage -le 1 ]; then
  local/prepare_data.sh
fi
