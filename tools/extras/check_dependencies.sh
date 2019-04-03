#!/usr/bin/env bash

CXX=${CXX:-g++}
status=0

# at some point we could try to add packages for Cywgin or macports(?) to this
# script.
redhat_packages=
debian_packages=
opensuse_packages=

function add_packages {
  redhat_packages="$redhat_packages $1";
  debian_packages="$debian_packages $2";
  opensuse_packages="$opensuse_packages $3";
}

if ! which which >&/dev/null; then
  echo "$0: which is not installed."
  add_packages which debianutils which
fi

COMPILER_VER_INFO=$($CXX --version 2>/dev/null)
case $COMPILER_VER_INFO in
  "")
    echo "$0: $CXX is not installed."
    echo "$0: You need g++ >= 4.8.3, Apple Xcode >= 5.0 or clang >= 3.3."
    add_packages gcc-c++ g++ gcc-c++
    status=1
    ;;
  "g++ "* )
    GCC_VER=$($CXX -dumpversion)
    GCC_VER_NUM=$(echo $GCC_VER | sed 's/\./ /g' | xargs printf "%d%02d%02d")
    if [ $GCC_VER_NUM -lt 40803 ]; then
        echo "$0: $CXX (g++-$GCC_VER) is not supported."
        echo "$0: You need g++ >= 4.8.3, Apple clang >= 5.0 or LLVM clang >= 3.3."
        status=1
    fi
    ;;
  "Apple LLVM "* )
    # See https://gist.github.com/yamaya/2924292
    CLANG_VER=$(echo $COMPILER_VER_INFO | grep version | sed "s/.*version \([0-9\.]*\).*/\1/")
    CLANG_VER_NUM=$(echo $COMPILER_VER_INFO | grep version | sed "s/.*clang-\([0-9]*\).*/\1/")
    if [ $CLANG_VER_NUM -lt 500 ]; then
        echo "$0: $CXX (Apple clang-$CLANG_VER) is not supported."
        echo "$0: You need g++ >= 4.8.3, Apple clang >= 5.0 or LLVM clang >= 3.3."
        status=1
    fi
    ;;
  "clang "* )
    CLANG_VER=$(echo $COMPILER_VER_INFO | grep version | sed "s/.*version \([0-9\.]*\).*/\1/")
    CLANG_VER_NUM=$(echo $CLANG_VER | sed 's/\./ /g' | xargs printf "%d%02d")
    if [ $CLANG_VER_NUM -lt 303 ]; then
        echo "$0: $CXX (LLVM clang-$CLANG_VER) is not supported."
        echo "$0: You need g++ >= 4.8.3, Apple clang >= 5.0 or LLVM clang >= 3.3."
        status=1
    fi
    ;;
  *)
    echo "$0: WARNING: unknown compiler $CXX."
    ;;
esac

if ! echo "#include <zlib.h>" | $CXX -E - >&/dev/null; then
  echo "$0: zlib is not installed."
  add_packages zlib-devel zlib1g-dev zlib-devel
fi

for f in make automake autoconf patch grep bzip2 gzip unzip wget git sox; do
  if ! which $f >&/dev/null; then
    echo "$0: $f is not installed."
    add_packages $f $f $f
  fi
done

if ! which libtoolize >&/dev/null && ! which glibtoolize >&/dev/null; then
  echo "$0: neither libtoolize nor glibtoolize is installed"
  add_packages libtool libtool libtool
fi

if ! which svn >&/dev/null; then
  echo "$0: subversion is not installed"
  add_packages subversion subversion subversion
fi

if ! which awk >&/dev/null; then
  echo "$0: awk is not installed"
  add_packages gawk gawk gawk
fi

pythonok=true
if ! which python2.7 >&/dev/null; then
  echo "$0: python2.7 is not installed"
  add_packages python2.7 python2.7
  pythonok=false
fi

if ! which python3 >&/dev/null; then
  echo "$0: python3 is not installed"
  add_packages python3 python3
  pythonok=false
fi

(
#Use a subshell so that sourcing env.sh does not have an influence on the rest of the script
[ -f ./env.sh ] && . ./env.sh
if $pythonok && ! which python2 >&/dev/null; then
  mkdir -p $PWD/python
  echo "$0: python2.7 is installed, but the python2 binary does not exist. Creating a symlink and adding this to tools/env.sh"
  ln -s $(which python2.7) $PWD/python/python2
  echo "export PATH=$PWD/python:\${PATH}" >> env.sh
fi

if [[ -f $PWD/python/.use_default_python && -f $PWD/python/python ]]; then
  rm $PWD/python/python
fi

if $pythonok && which python >&/dev/null && [[ ! -f $PWD/python/.use_default_python ]]; then
  version=`python 2>&1 --version | awk '{print $2}' `
  if [[ $version != "2.7"* ]] ; then
    echo "$0: WARNING python 2.7 is not the default python. We fixed this by adding a correct symlink more prominently on the path."
    echo "$0: If you really want to use python $version as default, add an empty file $PWD/python/.use_default_python and run this script again."
    mkdir -p $PWD/python
    ln -s $(which python2.7) $PWD/python/python
    echo "export PATH=$PWD/python:\${PATH}" >> env.sh
  fi
fi
)

printed=false

if which apt-get >&/dev/null && ! which zypper >/dev/null; then
  # if we're using apt-get [but we're not OpenSuse, which uses zypper as the
  # primary installer, but sometimes installs apt-get for some compatibility
  # reason without it really working]...
  if [ ! -z "$debian_packages" ]; then
    echo "$0: we recommend that you run (our best guess):"
    echo " sudo apt-get install $debian_packages"
    printed=true
    status=1
  fi
  if ! dpkg -l | grep -E 'libatlas3gf|libatlas3-base' >/dev/null; then
    echo "You should probably do: "
    echo " sudo apt-get install libatlas3-base"
    printed=true
  fi
elif which yum >&/dev/null; then
  if [ ! -z "$redhat_packages" ]; then
    echo "$0: we recommend that you run (our best guess):"
    echo " sudo yum install $redhat_packages"
    printed=true
    status=1
  fi
  if ! rpm -qa|  grep atlas >/dev/null; then
    echo "You should probably do something like: "
    echo "sudo yum install atlas.x86_64"
    printed=true
  fi
elif which zypper >&/dev/null; then
  if [ ! -z "$opensuse_packages" ]; then
    echo "$0: we recommend that you run (our best guess):"
    echo " sudo zypper install $opensuse_packages"
    printed=true
    status=1
  fi
  if ! zypper search -i | grep -E 'libatlas3|libatlas3-devel' >/dev/null; then
    echo "You should probably do: "
    echo "sudo zypper install libatlas3-devel"
    printed=true
  fi
fi

if [ ! -z "$debian_packages" ]; then
  # If the list of packages to be installed is nonempty,
  # we'll exit with error status.  Check this outside of
  # checking for yum or apt-get, as we want it to exit with
  # error even if we're not on Debian or red hat.
  status=1
fi


if [ $(pwd | wc -w) -gt 1 ]; then
  echo "*** $0: Warning: Kaldi scripts will fail if the directory name contains a space."
  echo "***  (it's OK if you just want to compile a few tools -> disable this check)."
  status=1;
fi

if which grep >&/dev/null && pwd | grep -E 'JOB|LMWT' >/dev/null; then
  echo "*** $0: Kaldi scripts will fail if the directory name contains"
  echo "***  either of the strings 'JOB' or 'LMWT'."
  status=1;
fi

if ! $printed && [ $status -eq 0 ]; then
  echo "$0: all OK."
fi

exit $status
