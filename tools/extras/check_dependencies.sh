#!/bin/bash

redhat_packages=
debian_packages=

function add_packages {
  redhat_packages="$redhat_packages $1";
  debian_packages="$debian_packages $2";
}

if ! which g++ >&/dev/null; then
  echo "$0: g++ is not installed."
  add_packages gcc-c++ g++
fi

if ! echo "#include <zlib.h>" | gcc -E - >&/dev/null; then
  echo "$0: zlib is not installed."
  add_packages zlib-devel zlib1g-dev
fi

for f in make automake libtool autoconf patch awk grep; do
  if ! which $f >&/dev/null; then
    echo "$0: $f is not installed."
    add_packages $f $f
  fi
done

if ! which svn >&/dev/null; then
  echo "$0: subversion is not installed"
  add_packages subversion subversion
fi

if ! which awk >&/dev/null; then
  echo "$0: awk is not installed"
  add_packages gawk gawk
fi


printed=false
status=0

if which apt-get >&/dev/null; then
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
  # Debian systems generally link /bin/sh to dash, which doesn't work
  # with some scripts as it doesn't expand x.{1,2}.y to x.1.y x.2.y
  if [ $(readlink /bin/sh) == "dash" ]; then
    echo "/bin/sh is linked to dash, and currently some of the scripts will not run"
    echo "properly.  We recommend to run:"
    echo " sudo ln -s -f dash /bin/sh"
    printed=true
  fi
fi

if which yum >&/dev/null; then
  if [ ! -z "$redhat_packages" ]; then
    echo "$0: we recommend that you run (our best guess):"
    echo " sudo yum install $redhat_packages"
    printed=true
    status=1
  fi
  if ! dpkg -l | grep atlas >/dev/null; then
    echo "You should probably do something like: "
    echo "sudo yum install atlas.x86_64"
    printed=true
  fi
fi


if ! $printed; then
  echo "$0: all OK."
fi
exit $status