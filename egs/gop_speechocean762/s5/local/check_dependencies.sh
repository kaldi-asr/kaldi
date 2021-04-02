#!/usr/bin/env bash

# Copyright 2015  Johns Hopkins University (Author: Jan Trmal <jtrmal@gmail.com>)
#           2021  Xiaomi Corporation (Author: Junbo Zhang)
# Apache 2.0

[ -f ./path.sh ] && . ./path.sh

command -v python3 >&/dev/null \
  || { echo  >&2 "python3 not found on PATH. You will have to install Python3, preferably >= 3.6"; exit 1; }

for package in kaldi_io sklearn imblearn seaborn; do
  python3 -c "import ${package}" 2> /dev/null
  if [ $? -ne 0 ] ; then
    echo >&2 "This recipe needs the package ${package} installed. Exit."
    exit 1
  fi
done

exit  0
