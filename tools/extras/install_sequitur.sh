#!/bin/bash
set -u
set -e
g2p_archive=g2p-r1668.tar.gz


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


[ ! -f $g2p_archive ] && wget http://www-i6.informatik.rwth-aachen.de/web/Software/$g2p_archive
tar xzf $g2p_archive
mv g2p sequitur


cd sequitur
patch  < ../extras/sequitur.patch
make
python setup.py install --prefix `pwd`

cd ../

(
  set +u
  [ ! -z ${SEQUITUR} ] && \
    echo >&2 "SEQUITUR variable is aleady defined. Undefining..." && \
    unset SEQUITUR

  [ -f ./env.sh ] && . ./env.sh

  [ ! -z ${SEQUITUR} ] && \
    echo >&2 "SEQUITUR config is already in env.sh" && exit

  wd=`pwd`
  wd=`readlink -f $wd`

  echo "export SEQUITUR=$wd/sequitur"
  echo "export PATH=\$PATH:\${SEQUITUR}/bin"
  echo "_site_packages=\`readlink -f \${SEQUITUR}/lib/python*/site-packages\`"
  echo "export PYTHONPATH=\$PYTHONPATH:\$_site_packages"
) >> env.sh

echo >&2 "Installation of SEQUITUR finished successfully"
echo >&2 "Please source the tools/env.sh in your path.sh to enable it"

