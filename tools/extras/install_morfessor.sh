#!/bin/bash
# Copyright  2017  Atlas Guide (Author : Lucas Jo)
#
# Apache 2.0
#

echo "#### installing morfessor"
dirname=morfessor
if [ ! -d ./$dirname ]; then
  mkdir -p ./$dirname
  git clone https://github.com/aalto-speech/morfessor.git morfessor ||
    {
      echo  >&2 "$0: Error git clone operation "
      echo  >&2 "  Failed in cloning the github repository (https://github.com/aalto-speech/morfessor.git)"
      exit
    }
fi

# env.sh setup
(
  set +u
  [ ! -z "${MORFESSOR}" ] && \
    echo >&2 "morfessor variable is aleady defined. undefining..." && \
    unset MORFESSOR

  [ -f ./env.sh ] && . ./env.sh

  [ ! -z "${MORFESSOR}" ] && \
    echo >&2 "MORFESSOR config is already in env.sh" && exit

  wd=`pwd`
  wd=`readlink -f $wd || pwd`

  echo "export MORFESSOR=\"$wd/morfessor\""
  echo "export PATH=\"\$PATH:\${MORFESSOR}/scripts\""
  echo "export PYTHONPATH=\"\${PYTHONPATH:-}:\$MORFESSOR\""
) >> env.sh

echo >&2 "installation of MORFESSOR finished successfully"
echo >&2 "please source tools/env.sh in your path.sh to enable it"
