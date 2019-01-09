#!/bin/bash
[ $# == 0 ] && echo "Usage: $0 src/[some-dir]/*.{h,cc}" && exit 1

# Let's run a set of in-place modifications by sed-commands,
for file in $@; do
  perl -i -pe 's/; \/\//;  \/\//' $file # '; //' -> ';  //'
  perl -i -pe 's/{ \/\//{  \/\//' $file # '{ //' -> '{  //'
  perl -i -pe 's/} \/\//}  \/\//' $file # '} //' -> '}  //'
  perl -i -pe 's/for(/for (/' $file     # 'for(' -> 'for ('
  perl -i -pe 's/if(/if (/' $file       # 'if(' -> 'if ('
  perl -i -pe 's/\s\s*$//' $file        # 'remove white-space at the end of lines'
done

