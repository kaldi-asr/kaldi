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
if ! $(python -c "import distutils.sysconfig" &> /dev/null); then
    echo "$0: WARNING: python library distutils.sysconfig not usable, this is necessary to figure out the path of Python.h." >&2
    echo "Proceeding with installation." >&2
else
  # get include path for this python version
  INCLUDE_PY=$(python -c "from distutils import sysconfig as s; print s.get_config_vars()['INCLUDEPY']")
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

command -v swig >/dev/null 2>&1 || {
  echo >&2 "$0: Error: I require swig but it's not installed.";
  echo >&2 "  Please install swig and run this script again. "
  exit 1;
}

if [ -d ./g2p ] || [ -d sequitur ] ; then
  echo  >&2 "$0: Warning: old installation of Sequitur found. You should manually"
  echo  >&2 "  delete the directories tools/sequitur and/or tools/g2p and "
  echo  >&2 "  edit the file tools/env.sh and remove manually all references to it"
  exit 1
fi

if [ ! -d ./sequitur-g2p ] ; then
  git clone https://github.com/sequitur-g2p/sequitur-g2p.git sequitur-g2p ||
  {
    echo  >&2 "$0: Warning: git clone operation ended unsuccessfully"
    echo  >&2 "  I will assume this is because you don't have https support"
    echo  >&2 "  compiled into your git "
    git clone git@github.com:sequitur-g2p/sequitur-g2p.git sequitur-g2p

    if [ $? -ne 0 ]; then
      echo  >&2 "$0: Error git clone operation ended unsuccessfully"
      echo  >&2 "  Clone the github repository (https://github.com/sequitur-g2p/sequitur-g2p.git)"
      echo  >&2 "  manually and re-run the script"
    fi
  }
else
  echo >&2 "$0: Updating the repository -- we will try to merge with local changes (if you have any)"
  (
    cd sequitur-g2p/
    git pull
    # this would work also, but would drop all local modifications
    #git fetch
    #git reset --hard origin/master
  ) || {
    echo >&2 "Failed to do git pull, delete the sequitur dir and run again";
    exit 1
  }
fi

(
cd sequitur-g2p

#we had some reports that the CPPFLAGS is needed under MacOS X but we could not
#reproduce it, actually, this, however, seems to work just fine for us
#the primary issue is that real GNU GCC does not accept that switch
#in addition, Apple fake g++ based on LLVM version 8.1 prints warning about
#the libstdc++ should no longer be used.
if (g++ --version 2>/dev/null | grep -s  "LLVM version 8.0" >/dev/null) ; then
  #Apple fake-g++
  make CXX=g++ CC=gcc CPPFLAGS="-stdlib=libstdc++"
else
  make CXX=g++ CC=gcc
fi

# the next two lines deal with the issue that the new setup tools
# expect the directory in which we will be installing to be visible
# as module directory to python
site_packages_dir=$(PYTHONPATH="" python -m site --user-site | grep -oE "lib.*")
SEQUITUR=$(pwd)/$site_packages_dir
# some bits of info to troubleshoot this in case people have problems
echo -n  >&2 "USER SITE: "; PYTHONPATH="" python -m site --user-site
echo >&2 "SEQUITUR_PACKAGE: ${site_packages_dir:-}"
echo >&2 "SEQUITUR: $SEQUITUR"
echo >&2 "PYTHONPATH: ${PYTHONPATH:-}"
mkdir -p $SEQUITUR
PYTHONPATH=${PYTHONPATH:-}:$SEQUITUR python setup.py install --prefix `pwd`
) || {
  echo >&2 "Problem installing sequitur!"
  exit 1
}

site_packages_dir=$(cd sequitur-g2p; find ./lib{,64} -type d -name site-packages | head -n 1)
(
  set +u
  [ ! -z "${SEQUITUR}" ] && \
    echo >&2 "SEQUITUR variable is aleady defined. Undefining..." && \
    unset SEQUITUR

  [ -f ./env.sh ] && . ./env.sh

  [ ! -z "${SEQUITUR}" ] && \
    echo >&2 "SEQUITUR config is already in env.sh" && exit

  wd=`pwd`
  wd=`readlink -f $wd || pwd`

  echo "export SEQUITUR=\"$wd/sequitur-g2p\""
  echo "export PATH=\"\$PATH:\${SEQUITUR}/bin\""
  echo "export PYTHONPATH=\"\${PYTHONPATH:-}:\$SEQUITUR/${site_packages_dir}\""
) >> env.sh

echo >&2 "Installation of SEQUITUR finished successfully"
echo >&2 "Please source tools/env.sh in your path.sh to enable it"
