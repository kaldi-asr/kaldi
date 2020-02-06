#!/usr/bin/env bash
# Copyright (c) 2015, Johns Hopkins University (Yenda Trmal <jtrmal@gmail.com>)
# License: Apache 2.0

# Begin configuration section.
# End configuration section

GIT=${GIT:-git}

set -e -o pipefail


errcho() { echo "$@" 1>&2; }

errcho "****() Installing IRSTLM"

if [ ! -d ./extras ]; then
  errcho "****** You are trying to install IRSTLM from the wrong directory.  You should"
  errcho "****** go to tools/ and type extras/install_irstlm.sh."
  exit 1
fi


if [ ! -d ./irstlm ] ; then
  if ! $GIT --version >&/dev/null ; then
    errcho "****() You need to have git installed"
    exit 1
  fi
  (
    $GIT clone https://github.com/irstlm-team/irstlm.git irstlm
  ) || {
    errcho "****() Error getting the IRSTLM sources. The server hosting it"
    errcho "****() might be down."
    exit 1
  }
else
  echo "****() Assuming IRSTLM is already installed. Please delete"
  echo "****() the directory ./irstlm if you need us to download"
  echo "****() the sources again."
  exit 0
fi

(
	cd irstlm || exit 1
  automake --version | grep 1.13.1 >/dev/null && \
         sed s:AM_CONFIG_HEADER:AC_CONFIG_HEADERS: <configure.in >configure.ac;

  patch -p1 < ../extras/irstlm.patch
  ./regenerate-makefiles.sh || ./regenerate-makefiles.sh

  ./configure --prefix `pwd`

	make; make install
) || {
  errcho "***() Error compiling IRSTLM. The error messages could help you "
  errcho "***() in figuring what went wrong."
  exit 1
}

(
  [ ! -z "${IRSTLM}" ] && \
    echo >&2 "IRSTLM variable is aleady defined. Undefining..." && \
    unset IRSTLM

  [ -f ./env.sh ] && . ./env.sh
  [ ! -z "${IRSTLM}" ] && \
    echo >&2 "IRSTLM config is already in env.sh" && exit

  wd=`pwd -P`

  echo "export IRSTLM=$wd/irstlm"
  echo "export PATH=\${PATH}:\${IRSTLM}/bin"
) >> env.sh

errcho "***() Installation of IRSTLM finished successfully"
errcho "***() Please source the tools/extras/env.sh in your path.sh to enable it"
