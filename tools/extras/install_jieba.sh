#!/usr/bin/env bash

# The script downloads and installs jieba

GIT=${GIT:-git}

set -e

# Make sure we are in the tools/ directory.
if [ `basename $PWD` == extras ]; then
  cd ..
fi

! [ `basename $PWD` == tools ] && \
   echo "You must call this script from the tools/ directory" && exit 1;

echo "Installing jieba"

if [ -d ./jieba ] ; then
  echo  >&2 "$0: Warning: old installation of jieba found. You should manually"
  echo  >&2 "  delete the directory tools/jieba and "
  echo  >&2 "  edit the file tools/env.sh and remove manually all references to it"
  exit 1
fi

if [ ! -d ./jieba ]; then
  $GIT clone https://github.com/fxsjy/jieba.git || exit 1;
fi

(
cd jieba
pyver=`python --version 2>&1 | sed -e 's:.*\([2-3]\.[0-9]\+\).*:\1:g'`
export PYTHONPATH=$PYTHONPATH:$PWD/lib/python${pyver}/site-packages/
# we have to create those dir, as the install target does not create it
mkdir -p $PWD/lib/python${pyver}/site-packages/
python setup.py install --prefix `pwd`
cd ..
)

lib_dir=./lib/
site_packages_dir=$(cd ./jieba; find $lib_dir -name "site-packages" -type d | head -n1)
(
  echo "export JIEBA=\"$PWD/jieba\""
  echo "export PYTHONPATH=\"\${PYTHONPATH:-}:\$JIEBA/${site_packages_dir}\""
) >> env.sh

echo >&2 "Installation of jieba finished successfully"
echo >&2 "Please source tools/env.sh in your path.sh to enable it"
