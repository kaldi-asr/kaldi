#!/usr/bin/env bash


# to be run from top level of repo.
find egs -maxdepth 7 \( -name '*.sh' -o -name '*.py' \) -exec cat {} \; | awk '{for (n=1;n<=NF;n++) seen[$n] = 1; } END{ for (k in seen) { print k; }}' > seen_tokens

for d in src/*bin; do
  if [ -d $d ] && [ -f $d/Makefile ]; then
    cat $d/Makefile | perl -ane ' while(<>){ s/\\\n//g; print; }' | grep -E '^BINFILES' | awk '{for(n=3;n<=NF;n++){print $n;}}';
  fi
done > all_binaries

for f in $(cat all_binaries); do if ! grep -q $f seen_tokens; then echo $f; fi; done
