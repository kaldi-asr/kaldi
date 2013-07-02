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

echo "Checking for libffi-dev"
exe_ffi=test_ffi
src_ffi=$exe_ffi.c
cat > $src_ffi <<CCODE
#include <stdio.h>
#include <ffi.h>
int main() {
ffi_cif cif; ffi_type *args[1]; void *values[1]; char *s; int rc;
/* Initialize the argument info vectors */
args[0] = &ffi_type_pointer;
values[0] = &s;
/* Initialize the cif */
if (ffi_prep_cif(&cif, FFI_DEFAULT_ABI, 1, &ffi_type_uint, args) == FFI_OK) {
   s = "Hello World!- YOU HAVE ffi INSTALLED";
   ffi_call(&cif, puts, &rc, values);
   /* rc now holds the result of the call to puts */
   /* values holds a pointer to the function's arg, so to
      call puts() again all we need to do is change the
      value of s */
   s = "FFI works!";
   ffi_call(&cif, puts, &rc, values);
 }
return 0;
}
CCODE
rm -f $exe_ffi  # clean previous attempts
gcc -o $exe_ffi $src_ffi -lffi  # build 
chmod u+x $exe_ffi  # make it executable (gcc usually does it too)

# checking the exit status = ffi installed?
./$exe_ffi  
if [ $? -ne 0 ] ; then 
    echo "You have not ffi installed!" 
    echo "On Debian/Ubuntu: sudo apt-get install libffi-dev"
    echo "Fedora: sudo yum libffi-devel"
    echo "Mac OS: brew install libffi"
    exit 1
fi


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

python_version=`python -c 'import sys; p1, p2, _, _ , _= sys.version_info;  print "python%d.%d" % (p1, p2)'`
new_ppath="$prefix/lib/$python_version/site-packages"
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
