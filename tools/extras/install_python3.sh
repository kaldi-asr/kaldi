#!/bin/bash

if hash python3 2> /dev/null; then
    echo 'python3 is already installed.';
    exit 0;
fi

if [ $(basename $(pwd)) != "tools" ]; then
  echo "$0: expected to be run from the tools directory."
  exit 1;
fi

py3dir="Python-3.3.5"

# Make sure the source is available
if [ ! -f $py3dir.tgz ]; then
    wget --no-check-certificate https://www.python.org/ftp/python/3.3.5/${py3dir}.tgz || exit 1;
fi

if [ ! -d $py3dir ]; then
    tar -xzf ${py3dir}.tgz || exit 1;
fi

# Exit if something went wrong to prevent calling make on other tools
cd $py3dir || exit 1;

./configure --prefix=$(pwd)

make

# Use altinstall as target so current python installations remain unchanged
make altinstall

cd ..
# link the python installation to somewhere in PATH, currently in utils/
ln -s $(pwd)/$py3dir/python ../egs/sprakbanken/s5/utils/python3
