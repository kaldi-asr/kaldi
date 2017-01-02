#!/bin/bash
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

if [ -d ./mmseg-1.3.0 ] ; then
  echo  >&2 "$0: Warning: old installation of mmseg found. You should manually"
  echo  >&2 "  delete the directory tools/mmseg and "
  echo  >&2 "  edit the file tools/env.sh and remove manually all references to it"
fi

if [ ! -d ./mmseg-1.3.0 ] ; then
  wget http://pypi.python.org/packages/source/m/mmseg/mmseg-1.3.0.tar.gz
  tar xf mmseg-1.3.0.tar.gz
fi

pyver=`python --version 2>&1 | sed -e 's:.*\([2-3]\.[0-9]\+\).*:\1:g'`
export PYTHONPATH=$PYTHONPATH:`pwd`/mmseg-1.3.0/lib/python${pyver}/site-packages
cd mmseg-1.3.0
mkdir -p lib/python${pyver}/site-packages
python setup.py build
python setup.py install --prefix `pwd`
cd ../

(
  set +u
  pyver=`python --version 2>&1 | sed -e 's:.*\([2-3]\.[0-9]\+\).*:\1:g'`
  wd=`pwd`

  [ -f ./env.sh ] && . ./env.sh

  echo "export PYTHONPATH=\$PYTHONPATH:$wd/mmseg-1.3.0/lib/python${pyver}/site-packages"
) >> env.sh

echo >&2 "Installation of mmseg finished successfully"
echo >&2 "Please source tools/env.sh in your path.sh to enable it"
