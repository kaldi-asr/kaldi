#!/usr/bin/env bash

#Replaces #!/bin/bash with #!/usr/bin/env bash to make the bash shell scripts more portable.

# To run this, cd to the top level of the repo and type
# misc/maintenance/fix_bash_shebang.sh

grep -rl --include=*.sh "^#\!/bin/bash$" . | xargs -I@ bash -c $'sed -i.BAK \'s:#!/bin/bash:#!/usr/bin/env bash:\' @ ; rm @.BAK'
