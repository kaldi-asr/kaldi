#!/bin/bash
[ $# == 0 ] && echo "Usage: $0 src/[some-dir]/*.{h,cc}" && exit 1

# Let's run a set of in-place modifications by sed-commands,
for file in $@; do
  sed -i 's/; \/\//;  \/\//' $file # '; //' -> ';  //'
  sed -i 's/{ \/\//{  \/\//' $file # '{ //' -> '{  //'
  sed -i 's/} \/\//}  \/\//' $file # '} //' -> '}  //'
  sed -i 's/for(/for (/' $file     # 'for(' -> 'for ('
  sed -i 's/if(/if (/' $file       # 'if(' -> 'if ('
  sed -i 's/\s\s*$//' $file        # 'remove white-space at the end of lines'
done

