#!/bin/bash

# We take into account dependency pointers optionally set in the environment.
# Typical usage shown below; any one can be safely left unset.
#   INCDIRS="~/xroot/usr/include"
#   LIBDIRS="~/xroot/usr/lib /usr/lib/openblas-base"
#   CXX=clang++-3.8
#   CFLAGS="-march=native -O2"
#   LDFLAGS="-llapack"

# Maximum make parallelism. Simply -j runs out of memory on Travis VM.
MAXPAR=6

# Directories with code that can be tested with Travis (space-separated)
TESTABLE_DIRS="src/"

# Run verbose (run and echo) and exit if failed.
runvx() {
  echo "\$ $@"
  eval "$@" || exit 1
}

# $(addsw -L foo bar) => "-Lfoo -Lbar".
addsw() {
  local v=() s=$1; shift;
  for d; do v+=("$s$d"); done
  echo ${v[@]};
}

# $(mtoken CXX gcc) => "CXX=gcc"; # $(mtoken CXX ) => "".
mtoken() { echo ${2+$1=\"$2\"}; }

# Print machine info and environment.
runvx uname -a
runvx env

# Check for changes in src/ only; report success right away if none.
# However, do run tests if TRAVIS_COMMIT_RANGE does not parse. This
# most likely means the branch was reset by --force; re-run tests then.
if git rev-parse "${TRAVIS_COMMIT_RANGE}" >/dev/null 2>&1 && \
   ! git diff --name-only "${TRAVIS_COMMIT_RANGE}" -- ${TESTABLE_DIRS} \
   .travis.yml tools/extras/travis_*.sh | read REPLY
then
  echo; echo "No changes outside ${TESTABLE_DIRS} in the commit" \
             "range ${TRAVIS_COMMIT_RANGE}; reporting success."
  exit 0;
fi

# Prepare environment variables
CF="\"$CFLAGS -g $(addsw -I $INCDIRS)\""
LDF="\"$LDFLAGS $(addsw -L $LIBDIRS)\""
CCC="$(mtoken CXX "$CXX")"

# Randomly choose between single and double precision
if [[ $(( RANDOM % 2 )) == 1 ]] ; then
  DPF="--double-precision=yes"
else
  DPF="--double-precision=no"
fi

echo "Building tools..." [Time: $(date)]
runvx cd tools
runvx make openfst "$CCC" CXXFLAGS="$CF" -j$MAXPAR
cd ..

echo "Building src..." [Time: $(date)]
runvx cd src
runvx "$CCC" CXXFLAGS="$CF" LDFLAGS="$LDF" ./configure --shared --use-cuda=no "$DPF" --mathlib=OPENBLAS --openblas-root="$XROOT/usr"
runvx make all -j$MAXPAR
runvx make ext -j$MAXPAR

echo "Running tests..." [Time: $(date)]
runvx make test -k -j$MAXPAR

echo "Done." [Time: $(date)]

#runvx make mklibdir base matrix -j$MAXPAR
#runvx make matrix/test
