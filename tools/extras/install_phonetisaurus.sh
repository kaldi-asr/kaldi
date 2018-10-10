#!/bin/bash
set -u
set -e


# Make sure we are in the tools/ directory.
if [ `basename $PWD` == extras ]; then
  cd ..
fi

! [ `basename $PWD` == tools ] && \
  echo "You must call this script from the tools/ directory" && exit 1;

# Install python-devel package if not already available
# first, makes sure distutils.sysconfig usable
# We are not currently compiling the bindings by default, but it seems
# worth it to keep this section as we do have them and they will
# probably be used.
if ! $(python -c "import distutils.sysconfig" &> /dev/null); then
    echo "$0: WARNING: python library distutils.sysconfig not usable, this is necessary to figure out the path of Python.h." >&2
    echo "Proceeding with installation." >&2
else
  # get include path for this python version
  INCLUDE_PY=$(python -c "from distutils import sysconfig as s; print(s.get_python_inc())")
  if [ ! -f "${INCLUDE_PY}/Python.h" ]; then
      echo "$0 : ERROR: python-devel/python-dev not installed" >&2
      if which yum >&/dev/null; then
        # this is a red-hat system
        echo "$0: we recommend that you run (our best guess):"
        echo " sudo yum install python-devel"
      fi
      if which apt-get >&/dev/null; then
        # this is a debian system
        echo "$0: we recommend that you run (our best guess):"
        echo " sudo apt-get install python-dev"
      fi
      exit 1
  fi
fi


if [ ! -d ./phonetisaurus-g2p ] ; then
  git clone https://github.com/AdolfVonKleist/Phonetisaurus.git phonetisaurus-g2p ||
  {
    echo  >&2 "$0: Warning: git clone operation ended unsuccessfully"
    echo  >&2 "  I will assume this is because you don't have https support"
    echo  >&2 "  compiled into your git "
    git clone http://github.com/AdolfVonKleist/Phonetisaurus.git phonetisaurus-g2p

    if [ $? -ne 0 ]; then
      echo  >&2 "$0: Error git clone operation ended unsuccessfully"
      echo  >&2 "  Clone the github repository (https://github.com/AdolfVonKleist/Phonetisaurus.git)"
      echo  >&2 "  manually make and install in accordance with directions."
    fi
  }
fi

(
    export TOOLS=${PWD}
    cd phonetisaurus-g2p
    #checkout the current kaldi tag
    git checkout -b kaldi kaldi
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
