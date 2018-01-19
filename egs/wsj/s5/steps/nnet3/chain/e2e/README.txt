The scripts related to end2end chain training are in this directory
Currently it has two scripts:

** prepare_e2e.sh which is almost equivalent
to regular chain's build-tree.sh (i.e. it creates the tree and
the transition-model) except it does not require any previously
trained models (in other terms, it does what stages -3 and -2
of steps/train_mono.sh do).

** train_e2e.py: this is very similar to chain/train.py but
with fewer stages (e.g. it does not compute the preconditioning matrix)
