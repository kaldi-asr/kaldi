#!/usr/bin/env bash

set -e

if [ $# -ne 1 ]; then
    echo "The scripts checks whether requested rnnlm binary exists in tools/<rnnlm_ver>/rnnlm"
    echo
    echo "Usage: $0 <rnnlm_ver>"
    exit 1
fi

rnnlm_ver=$1
rnnlm_path="$(readlink -f "$(dirname "$0")/../")/$rnnlm_ver/rnnlm"
scriptname="$(basename "$0")"

if [ -f "$rnnlm_path" ]; then
    echo "$scriptname: Found binary $rnnlm_path"
else
    if [ $rnnlm_ver == "faster-rnnlm" ]; then
        echo "$scriptname: ERROR Faster RNNLM is not installed. Use extras/install_faster_rnnlm.sh to install it"
    elif [ $rnnlm_ver == "rnnlm-0.??" ]; then
        echo "$scriptname: ERROR Class based RNNLM is not installed. Use extras/install_mikolov_rnnlm.sh to install it"
    else
        echo "$scriptname: ERROR Cannot find $rnnlm_path. Neither know how to install it"
    fi
    exit 1
fi
