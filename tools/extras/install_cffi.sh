#!/bin/bash
#Copyright 2013 Ufal MFF UK; Ondrej Platek
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#http://www.apache.org/licenses/LICENSE-2.0
#
#THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
#WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
#MERCHANTABLITY OR NON-INFRINGEMENT.
#See the Apache 2 License for the specific language governing permissions and
#limitations under the License.
#
# This script is attempting to install cffi a Python/C interface
# Cffi is used to call Kaldi from Python
# See the documentation at http://cffi.readthedocs.org/en/latest/
# This script is trying to install cffi 0.6(now the latest version
# Tell us if you need higher version(if it exists).
#
# Also dependencies are installed. Namely:
# * pycparser >= 2.06: http://code.google.com/p/pycparser/
# * py.test
# 
# NOT INSTALLED DEPENDENCIES (We are letting you to install it!):
# * Cpython 2.6, 2.7 or PyPy(not tested for Kaldi setup) 
# * python-dev (Python headers) and libffi-dev (ffi C library)
# * a C compiler is required to use CFFI during development, but not to run correctly-installed programs that use CFF

echo "**** Installing Cffi and dependencies"

echo "Checking for Python-Dev"
# copied from http://stackoverflow.com/questions/4848566/check-for-existence-of-python-dev-files-from-bash-script
if [ ! -e $(python -c 'from distutils.sysconfig import get_makefile_filename as m; print m()') ]; then 
    echo "On Debian/Ubuntu like system install by 'sudo apt-get python-dev' package."
    echo "On Fedora by 'yum install python-devel'"
    echo "On Mac OS X by 'brew install python'"
    echo "Or obtain Python headers other way!"
    exit 1
fi

echo "Check that you have python-libffi-dev YOURSELF!"
echo "I do not know how to check it effectively!"
echo "Example commands how to intall ffi library"
echo "Debian/Ubuntu: sudo apt-get install libffi-dev"
echo "Fedora: sudo yum libffi-devel"
echo "Mac OS: brew install libffi"

# names of the extracted directories
cffiname=cffi-0.6
pycparsername=pycparser-release_v2.09.1
pytestname=pytest-2.3.5

# helper function  
function downloader {
    file=$1; url=$2;
    if [ ! -e "$file" ]; then
        echo "Could not find $file" 
        echo "Trying to download it via wget!"
        
        wget --version  >/dev/null 2>&1 || \
            { echo "This script requires you to first install wget"
            echo "You can also just download $file from $url"
            exit 1; }

       wget --no-check-certificate -T 10 -t 3 $url

       if [ ! -e $file ]; then
            echo "Download of $file - failed!"
            echo "Aborting script. Please download and install $file manually!"
        exit 1;
       fi
    fi
}

echo Downloading and extracting cffi
cffitar=$cffiname.tar.gz
cffiurl=http://pypi.python.org/packages/source/c/cffi/cffi-0.6.tar.gz
downloader $cffitar $cffiurl
tar -xovzf $cffitar || exit 1

echo Downloading and extracting pycparser
pycparsertar=release_v2.09.1.tar.gz
pycparserurl=https://github.com/eliben/pycparser/archive/release_v2.09.1.tar.gz
downloader $pycparsertar $pycparserurl
tar -xovzf $pycparsertar || exit 1

echo Downloading and extracting pytest
pytesttar=pytest-2.3.5.tar.gz
pytesturl=https://pypi.python.org/packages/source/p/pytest/pytest-2.3.5.tar.gz
downloader $pytesttar $pytesturl
tar -xovzf $pytesttar || exit 1

# Installing
prefix="$PWD/python"

new_ppath="$prefix/lib/python2.7/site-packages"
mkdir -p "$new_ppath"
export PYTHONPATH="$PYTHONPATH:$new_ppath"
echo; echo "Adding the $new_ppath to PYTHONPATH"
echo "DO THE SAME IN YOUR PERMANENT SETTINGS TO USE THE CFFI REGULARLY!"; echo


echo "*******Installing $pytestname"
pushd $pytestname

new_path="$prefix/bin"
export PATH="$PATH:$new_path"
echo "\nAdding the $new_path to PATH so I can launch the pytest"
echo "DO THE SAME IN YOUR PERMANENT SETTINGS TO USE THE pytest REGULARLY!\n"

python setup.py install --prefix="$prefix" || exit 1
popd


echo "*******Installing $pycparsername"
pushd $pycparsername
python setup.py install --prefix="$prefix" || exit 1
popd


echo "*******Installing $cffiname"
pushd $cffiname
# FIXME check the depencies 
python setup.py install --prefix="$prefix" || exit 1
popd


echo "****** Last check "
if [ ! -e $(python -c 'import cffi') ]; then 
    echo "Installation failed. Please download and install $file manually!"
fi

echo; echo SUCCESS ; echo
echo "ADD the"; echo "$new_ppath"; echo "to PYTHONPATH for using cffi regularly!"; echo
