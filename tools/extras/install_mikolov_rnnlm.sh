#!/bin/bash

set -e

if [ $# -ne 1 ]; then
    echo "Download and install given rnnlm version from rnnlm.org"
    echo
    echo "Usage: $0 <rnnlm_ver> # e.g. $0 rnnlm-0.3e"
    exit 1
fi

rnnlm_ver=$1
tools_dir="$(readlink -f "$(dirname "$0")/../")"

if [ "$(basename "$tools_dir")" != "tools" ]; then
    echo "Cannot find tools/ dir. Am I in tools/extras?"
    exit 1
fi

cd $tools_dir
echo Downloading and installing the rnnlm tools
# http://www.fit.vutbr.cz/~imikolov/rnnlm/$rnnlm_ver.tgz
arc_file="$rnnlm_ver.tgz"
if [ ! -f "$arc_file" ]; then
    wget "http://www.fit.vutbr.cz/~imikolov/rnnlm/$rnnlm_ver.tgz" -O "$arc_file" || exit 1;
fi
mkdir $rnnlm_ver
cd $rnnlm_ver
tar -xvzf ../$rnnlm_ver.tgz || exit 1;
patch  < ../extras/mikolov_rnnlm.patch
make CC=g++ || exit 1;
echo Done making the rnnlm tools
