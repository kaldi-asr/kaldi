#!/usr/bin/env bash

GIT=${GIT:-git}

set -u
set -e


# Make sure we are in the tools/ directory.
if [ `basename $PWD` == extras ]; then
  cd ..
fi

! [ `basename $PWD` == tools ] && \
  echo "You must call this script from the tools/ directory" && exit 1;


if [ ! -d ./phonetisaurus-g2p ] ; then
  $GIT clone https://github.com/danijel3/Phonetisaurus.git phonetisaurus-g2p ||
  {
    echo  >&2 "$0: Error git clone operation ended unsuccessfully"
    echo  >&2 "  Clone the github repository (https://github.com/danijel3/Phonetisaurus.git)"
    echo  >&2 "  manually make and install in accordance with directions."
  }
fi

(
    export TOOLS=${PWD}
    cd phonetisaurus-g2p
    #checkout the current kaldi tag
    $GIT checkout -b kaldi kaldi
    ./configure --with-openfst-includes=${TOOLS}/openfst/include --with-openfst-libs=${TOOLS}/openfst/lib
    make
)

(
  set +u
  [ ! -z "${PHONETISAURUS}" ] && \
    echo >&2 "PHONETISAURUS variable is aleady defined. Undefining..." && \
    unset PHONETISAURUS

  [ -f ./env.sh ] && . ./env.sh

  [ ! -z "${PHONETISAURUS}" ] && \
    echo >&2 "PHONETISAURUS config is already in env.sh" && exit

  wd=`pwd`
  wd=`readlink -f $wd || pwd`

  echo "export PHONETISAURUS=\"$wd/phonetisaurus-g2p\""
  echo "export PATH=\"\$PATH:\${PHONETISAURUS}:\${PHONETISAURUS}/src/scripts\""
) >> env.sh

echo >&2 "Installation of PHONETISAURUS finished successfully"
echo >&2 "Please source tools/env.sh in your path.sh to enable it"
echo >&2 "NOTE: only the C++ binaries are compiled by default."
echo >&2 " see the README.md file for details on installing the"
echo >&2 " optional python bindings and supplementary scripts."
