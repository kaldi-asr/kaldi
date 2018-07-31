#!/bin/bash
# Copyright (c) 2016, Johns Hopkins University (Yenda Trmal <jtrmal@gmail.com>)
# License: Apache 2.0

# Begin configuration section.
# End configuration section

#echo >&2 "$0 $@"  # Print the command line for logging
if [ $# != 2 ] ; then
  echo >&2 "Usage: $0  <first-dir> <second-dir>"
  echo >&2 " e.g.: $0 exp/nnet3/extractor exp/nnet3/ivectors_dev10h.pem"
fi

dir_a=$1
dir_b=$2

id_a=$(steps/nnet2/get_ivector_id.sh $dir_a)
ret_a=$?
id_b=$(steps/nnet2/get_ivector_id.sh $dir_b)
ret_b=$?

if [ ! -z "$id_a" ] && [ ! -z "${id_b}" ] ; then
  if [ "${id_a}" == "${id_b}" ]; then
    exit 0
  else
    echo >&2 "$0: ERROR: iVector id ${id_a} in $dir_a and the iVector id ${id_b} in $dir_b do not match"
    echo >&2 "$0: ERROR: that means that the systems are not compatible."
    exit 1
  fi
elif [ -z "$id_a" ] && [ -z "${id_b}" ] ; then
    echo >&2 "$0: WARNING: The directories do not contain iVector ID."
    echo >&2 "$0: WARNING: That means it's you who's reponsible for keeping "
    echo >&2 "$0: WARNING: the directories compatible"
    exit 0
else
    echo >&2 "$0: WARNING: One of the directories do not contain iVector ID."
    echo >&2 "$0: WARNING: That means it's you who's reponsible for keeping "
    echo >&2 "$0: WARNING: the directories compatible"
    exit 0
fi
