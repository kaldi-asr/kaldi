#!/usr/bin/env bash

# run this from the top level of the repo, as
# misc/maintenance/check_usage_messages.sh

set -e

cd src

echo "Any errors reported below must be fixed manually."
grep 'Usage:' *bin/*.cc | \
    perl -ane '@A = split; $path =$A[0]; $A[0] =~ s|.+/(.+).cc:|$1|; if ($A[0] ne $A[2]) { print "$path: $A[0] ne $A[2]\n"; } '

