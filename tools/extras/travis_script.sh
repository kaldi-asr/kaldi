#!/bin/bash

# We take into account dependency pointers optionally set in the environment.
# Typical usage shown below; any one can be safely left unset.
#   INCDIRS="~/xroot/usr/include"
#   LIBDIRS="~/xroot/usr/lib /usr/lib/openblas-base"
#   CXX=clang++-3.8
#   CFLAGS="-march=native -O2"
#   LDFLAGS="-llapack"
#
# The CI_TARGETS variable is set in Travis environment and passed on as a list
# of build targets to make (for making different things in separate jobs).

# Maximum make parallelism. Travis VMs have 2 cores, so a value over 3 or 4
# would probably only cause context switching overhead.
MAXPAR=4

# Directories with code that can be tested with Travis (space-separated).
TESTABLE_DIRS="src/"

# Run verbose (run and echo) and exit if failed.
runvx() {
  local cmd=$(printf ' %q' "$@"); cmd=${cmd:1}
  echo "\$ $cmd"
  eval -- "$cmd" || exit 1
}

# $(addsw -L foo bar) => "-Lfoo -Lbar".
addsw() {
  local v=() s=$1; shift;
  for d; do v+=("$s$d"); done
  echo ${v[@]};
}

# $(mtoken CXX "ccache gcc") => 'CXX=ccache gcc'; $(mtoken CXX ) => ''.
mtoken() { echo ${2+$1=$2}; }

# Print machine info and environment.
runvx uname -a
runvx env

# Check for changes in interesting files, normally sources and CI glue
# scripts, and report success right away if none.  However, do run tests
# if TRAVIS_COMMIT_RANGE does not parse. This most likely means the branch
# was reset by --force, and any file could have changed.
if git rev-parse "${TRAVIS_COMMIT_RANGE}" >/dev/null 2>&1 && \
   ! git diff --name-only "${TRAVIS_COMMIT_RANGE}" -- ${TESTABLE_DIRS} \
   .travis.yml tools/extras/travis_*.sh | read REPLY
then
  echo; echo "No changes outside ${TESTABLE_DIRS} in the commit" \
             "range ${TRAVIS_COMMIT_RANGE}; reporting success."
  exit 0;
fi

# Prepare environment variables.
CF="$CFLAGS -g $(addsw -I $INCDIRS)"
LDF="$LDFLAGS $(addsw -L $LIBDIRS)"
CCC=$(mtoken CXX "$CXX")

# TODO(kkm): Disabling single/double. If needed, use separate Travis jobs.
# Randomly choose between single and double precision
#if [[ $(( RANDOM % 2 )) == 1 ]] ; then
#  DPF="--double-precision=yes"
#else
  DPF="--double-precision=no"
#fi

echo "Building tools..." [Time: $(date)]
runvx cd tools
runvx make -j$MAXPAR openfst "$CCC" CXXFLAGS="$CF" \
      OPENFST_CONFIGURE="--disable-static --enable-shared --disable-bin --disable-dependency-tracking"
runvx make -j$MAXPAR cub "$CCC" CXXFLAGS="$CF"
cd ..

runvx cd src
runvx touch base/.depend.mk  # Fool make depend into skipping the dependency step.
runvx touch .short_version   # Make version short, or else ccache will miss everything.
runvx "$CCC" CXXFLAGS="$CF" LDFLAGS="$LDF" ./configure --shared --use-cuda=no "$DPF" --mathlib=OPENBLAS --openblas-root="$XROOT/usr"
runvx make -j$MAXPAR $CI_TARGETS CI_NOLINKBINARIES=1

# Travis has a 10k line log limit, so use smaller CI_TARGETS when logging.
if [ -r "$CCACHE_LOGFILE" ]; then
    runvx cat "$CCACHE_LOGFILE"
fi

echo "Done." [Time: $(date)]
