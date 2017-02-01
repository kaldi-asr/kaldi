#!/bin/bash

. ./cmd.sh

TYPE=$1
LANGDIR=$2
MODELDIR=$3
DEVDIR=$4
TRANSFORMDIR=$5

echo "$@"

if [ "$1" == "SI" ]; then
  utils/mkgraph.sh $LANGDIR $MODELDIR $MODELDIR/graph		|| exit 1
  steps/decode.sh --nj 20 --cmd "$decode_cmd" \
	  $MODELDIR/graph $DEVDIR $MODELDIR/decode || exit 1
elif [ "$1" == "FMLLR" ]; then
  utils/mkgraph.sh $LANGDIR $MODELDIR $MODELDIR/graph		|| exit 1
  steps/decode_fmllr.sh --config conf/decode.config --nj 20 --cmd "$decode_cmd" \
	  $MODELDIR/graph $DEVDIR $MODELDIR/decode || exit 1
fi


