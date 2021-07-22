#!/bin/bash

# To be sourced by another script

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <argument>" >&2
    echo " e.g.: $0 --data-dir" >&2
fi

key=$1

name=$(sed -e s/^--// -e s/-/_/g <<< "$key")

if eval '[ -z "$'$name'" ]'; then
    echo "$0: option $key is required" >&2
    echo >&2
    echo "$help_message" >&2
    exit 1
fi

