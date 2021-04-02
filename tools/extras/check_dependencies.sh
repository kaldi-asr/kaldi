#!/usr/bin/env bash

CXX=${CXX:-g++}
CXXFLAGS=${CXXFLAGS}
status=0

# at some point we could try to add packages for Cywgin or macports(?) to this
# script.
redhat_packages=
debian_packages=
opensuse_packages=

function add_packages {
  redhat_packages="$redhat_packages $1"
  debian_packages="$debian_packages ${2:-$1}"
  opensuse_packages="$opensuse_packages ${3:-$1}"
}

function have { type -t "$1" >/dev/null; }

compiler_ver_info=$($CXX --version 2>/dev/null)
case $compiler_ver_info in
  "")
    echo "$0: Compiler '$CXX' is not installed."
    echo "$0: You need g++ >= 4.8.3, Apple Xcode >= 5.0 or clang >= 3.3."
    add_packages gcc-c++ g++
    status=1
    ;;
  *"c++ "* | "g++ "* )
    gcc_ver=$($CXX -dumpversion)
    gcc_ver_num=$(echo $gcc_ver | sed 's/\./ /g' | xargs printf "%d%02d%02d")
    if [ $gcc_ver_num -lt 40803 ]; then
        echo "$0: Compiler '$CXX' (g++-$gcc_ver) is not supported."
        echo "$0: You need g++ >= 4.9.1, Apple clang >= 5.0 or LLVM clang >= 3.3."
        status=1
    fi
    ;;
  "Apple LLVM "* )
    # See https://gist.github.com/yamaya/2924292
    clang_ver=$(echo $compiler_ver_info | grep version | sed "s/.*version \([0-9\.]*\).*/\1/")
    clang_ver_num=$(echo $compiler_ver_info | grep version | sed "s/.*clang-\([0-9]*\).*/\1/")
    if [ $clang_ver_num -lt 500 ]; then
        echo "$0: Compiler '$CXX' (Apple clang-$clang_ver) is not supported."
        echo "$0: You need g++ >= 4.8.3, Apple clang >= 5.0 or LLVM clang >= 3.3."
        status=1
    fi
    ;;
  "clang "* | "Apple clang "* )
    clang_ver=$(echo $compiler_ver_info | grep version | sed "s/.*version \([0-9\.]*\).*/\1/")
    clang_ver_num=$(echo $clang_ver | sed 's/\./ /g' | xargs printf "%d%02d")
    if [ $clang_ver_num -lt 303 ]; then
        echo "$0: Compiler '$CXX' (LLVM clang-$clang_ver) is not supported."
        echo "$0: You need g++ >= 4.8.3, Apple clang >= 5.0 or LLVM clang >= 3.3."
        status=1
    fi
    ;;
  *)
    echo "$0: WARNING: unknown compiler $CXX."
    ;;
esac

# Cannot check this without a compiler.
if have "$CXX" && ! echo "#include <zlib.h>" | $CXX $CXXFLAGS -E - &>/dev/null; then
  echo "$0: zlib is not installed."
  add_packages zlib-devel zlib1g-dev
fi

for f in make automake autoconf patch grep bzip2 gzip unzip wget git sox; do
  if ! have $f; then
    echo "$0: $f is not installed."
    add_packages $f
  fi
done

if ! have gfortran; then
  echo "$0: gfortran is not installed"
  add_packages gcc-gfortran gfortran
fi

if ! have libtoolize && ! have glibtoolize; then
  echo "$0: neither libtoolize nor glibtoolize is installed"
  add_packages libtool
fi

if ! have svn; then
  echo "$0: subversion is not installed"
  add_packages subversion
fi

if ! have awk; then
  echo "$0: awk is not installed"
  add_packages gawk
fi

pythonok=true
if ! have python2.7; then
  echo "$0: python2.7 is not installed"
  add_packages python27 python2.7
  pythonok=false
fi

if ! have python3; then
  echo "$0: python3 is not installed"
  add_packages python3
  pythonok=false
fi

(
#Use a subshell so that sourcing env.sh does not have an influence on the rest of the script
[ -f ./env.sh ] && . ./env.sh
if $pythonok && ! have python2; then
  mkdir -p $PWD/python
  echo "$0: python2.7 is installed, but the python2 binary does not exist." \
       "Creating a symlink and adding this to tools/env.sh"
  ln -s $(command -v python2.7) $PWD/python/python2
  echo "export PATH=$PWD/python:\${PATH}" >> env.sh
fi

if [[ -f $PWD/python/.use_default_python && -f $PWD/python/python ]]; then
  rm $PWD/python/python
fi

if $pythonok && have python && [[ ! -f $PWD/python/.use_default_python ]]; then
  version=$(python 2>&1 --version | awk '{print $2}')
  if [[ $version != "2.7"* ]] ; then
    echo "$0: WARNING python 2.7 is not the default python. We fixed this by" \
         "adding a correct symlink more prominently on the path."
    echo " ... If you really want to use python $version as default, add an" \
         "empty file $PWD/python/.use_default_python and run this script again."
    mkdir -p $PWD/python
    ln -s $(command -v python2.7) $PWD/python/python
    echo "export PATH=$PWD/python:\${PATH}" >> env.sh
  fi
fi
)

mathlib_missing=false
case $(uname -m) in
  x86_64)  # Suggest MKL on an Intel64 system (not supported on i?86 hosts).
    # Respect user-supplied MKL_ROOT environment variable.
    MKL_ROOT="${MKL_ROOT:-/opt/intel/mkl}"
       # Check the well-known mkl.h file location.
    if ! [[ -f "${MKL_ROOT}/include/mkl.h" ]] &&
       # Ubuntu 20+ has an MKL package
       ! pkg-config mkl-dynamic-lp64-seq --exists &>/dev/null; then
      echo "$0: Intel MKL does not seem to be installed."
      if [[ $(uname) = Linux ]]; then
        echo $' ... Run extras/install_mkl.sh to install it. Some distros' \
             $'(e.g., Ubuntu 20.04) provide\n ... a version of MKL via'    \
             $'the package manager, but verify that it is up-to-date.'
      else
        echo $' ... Download the installer package for your system from:'  \
             $'\n ...   https://software.intel.com/mkl/choose-download'
      fi
      mathlib_missing=true
    fi
      ;;
  *)  # Suggest OpenBLAS on other hardware.
    if [ ! -f $(pwd)/OpenBLAS/install/include/openblas_config.h ] &&
         ! echo '#include <openblas_config.h>' |
            $CXX -I $(pwd)/OpenBLAS/install/include -E - &>/dev/null; then
      echo "$0: OpenBLAS not detected. Run extras/install_openblas.sh
 ... to compile it for your platform, or configure with --openblas-root= if you
 ... have it installed in a location we could not guess. Note that packaged
 ... library may be significantly slower and/or older than the one the above
 ... would build."
      mathlib_missing=true
    fi
      ;;
esac
$mathlib_missing &&
  echo "\
 ... You can also use other matrix algebra libraries. For information, see:
 ...   http://kaldi-asr.org/doc/matrixwrap.html"

# Report missing programs and libraries.
if [ -n "$debian_packages" ]; then
  install_pkg_command=$(
    # Guess package manager from user's distribution type. Use a subshell
    # because we are potentially importing a lot of dirt here.
    eval $(grep 2>/dev/null ^ID /etc/os-release) 2>/dev/null
    for rune in ${ID-} ${ID_LIKE-}; do
      # The case '(pattern)' syntax is necessary in subshell for bash 3.x.
      case $rune in
        (rhel|centos|redhat) echo "yum install $redhat_packages"; break;;
        (fedora) echo "dnf install $redhat_packages"; break;;
        (suse) echo "zypper install $opensuse_packages"; break;;
        (debian) echo "apt-get install $debian_packages"; break;;
      esac
    done
  )

  # Print the suggestion to install missing packages.
  if [ -n "$install_pkg_command" ]; then
    echo "$0: Some prerequisites are missing; install them using the command:"
    echo "  sudo" $install_pkg_command
  else
    echo "$0: The following prerequisites are missing; install them first:"
    echo "  " $debian_packages
  fi
  status=1
fi

if [ $(pwd | wc -w) -gt 1 ]; then
  echo "*** $0: Warning: Kaldi scripts will fail if the directory name contains a space."
  echo "***  (it's OK if you just want to compile a few tools -> disable this check)."
  status=1
fi

if pwd | grep -E 'JOB|LMWT' >/dev/null; then
  echo "*** $0: Kaldi scripts will fail if the directory name contains"
  echo "***  either of the strings 'JOB' or 'LMWT'."
  status=1
fi

if ! $mathlib_missing && [ $status -eq 0 ]; then
  echo "$0: all OK."
fi

exit $status
