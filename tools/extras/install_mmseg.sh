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

if [ -d ./mmseg-1.3.0 ] ; then
  echo  >&2 "$0: Warning: old installation of mmseg found. You should manually"
  echo  >&2 "  delete the directory tools/mmseg and "
  echo  >&2 "  edit the file tools/env.sh and remove manually all references to it"
  exit 1
fi

if [ ! -d ./mmseg-1.3.0 ] ; then
  wget http://pypi.python.org/packages/source/m/mmseg/mmseg-1.3.0.tar.gz
  tar xf mmseg-1.3.0.tar.gz
fi

(
cd mmseg-1.3.0
pyver=`python --version 2>&1 | sed -e 's:.*\([2-3]\.[0-9]\+\).*:\1:g'`
export PYTHONPATH=$PYTHONPATH:$PWD/lib/python${pyver}/site-packages/:$PWD/lib64/python${pyver}/site-packages/
# we have to create those dir, as the install target does not create it
mkdir -p $PWD/lib/python${pyver}/site-packages/
mkdir -p $PWD/lib64/python${pyver}/site-packages/
python setup.py build
python setup.py install --prefix `pwd`
)

## we first find the mmseg.py file (the module name which will be imported,
## so that should be pretty reliable) and then we work out the location of
## the site-packages directory (typically it would be one level up from
## the location of the mmseg.py file but using find seems more reliable
mmseg_file_lib=$(find ./mmseg-1.3.0/lib/ -type f -name mmseg.py | head -n1)
mmseg_file_lib64=$(find ./mmseg-1.3.0/lib64/ -type f -name mmseg.py | head -n1)
if [ ! -z ${mmseg_file_lib+x} ]; then
  lib_dir=./lib/
elif [ ! -z ${mmseg_file_lib64+x} ]; then
  lib_dir=./lib64/
else
  echo >&2 "$0: ERROR: Didn't find ./mmseg-1.3.0/lib/ or ./mmseg-1.3.0/lib64/"
  echo >&2 "  Perhaps your python or system installs python modules into"
  echo >&2 "  a different dir or some other unknown issues arised. Review the output"
  echo >&2 "  of the script and try to figure out what went wrong."
  exit 1
fi

site_packages_dir=$(cd ./mmseg-1.3.0; find $lib_dir -name "site-packages" -type d | head -n1)
(
  echo "export MMSEG=\"$PWD/mmseg-1.3.0\""
  echo "export PYTHONPATH=\"\${PYTHONPATH:-}:\$MMSEG/${site_packages_dir}\""
) >> env.sh

echo >&2 "Installation of mmseg finished successfully"
echo >&2 "Please source tools/env.sh in your path.sh to enable it"
