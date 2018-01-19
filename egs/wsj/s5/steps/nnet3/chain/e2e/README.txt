The scripts related to end2end chain training are in this directory
Currently it has 3 scripts:

** prepare_e2e.sh which is almost equivalent
to regular chain's build-tree.sh (i.e. it creates the tree and
the transition-model) except it does not require any previously
trained models (in other terms, it does what stages -3 and -2
of steps/train_mono.sh do).

** get_egs_e2e.sh: this is simlilar to chain/get_egs.sh except it
uses training FSTs (instead of lattices) to generate end2end egs.

** train_e2e.py: this is very similar to chain/train.py but
with fewer stages (e.g. it does not compute the preconditioning matrix)


For details please see the comments at top of local/chain/e2e/run_flatstart_*.sh
and also src/chain/chain-generic-numerator.h.
