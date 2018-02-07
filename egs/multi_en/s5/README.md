This is a WIP **English LVCSR recipe** that trains on data from multiple corpora. By default, it uses all of the following:
* Fisher (1761 hours)
* Switchboard (317 hours)
* WSJ (81 hours)
* HUB4 (1996 & 1997) English Broadcast News (75 + 72 hours)
* TED-LIUM (118 hours)
* Librispeech (960 hours)

It is possible to add or remove datasets as necessary.

This recipe was developed by Allen Guo (ICSI), Korbinian Riedhammer (Remeeting) and Xiaohui Zhang (JHU). The original spec for the recipe is at [#699](https://github.com/kaldi-asr/kaldi/issues/699).

## Recipe features

### Recipe variants

To make it as easy as possible to extend and modify this example, we provide the ability to create separate recipe variants. These variants are named `multi_a`, `multi_b`, etc. For instance, you could have

* `multi_a` = default
  * Bootstrap GMM-HMM on WSJ SI-84
  * Train SAT system with AMI/Fisher/Librispeech/SWBD/Tedlium/WSJ nnet3 model (no AMI-SDM)
  * Train tdnn (ivector) system on top of that
* `multi_b` = your first experiment
  * Train monophone model on SWBD
  * Then add in WSJ SI-284 for remaining GMM-HMM steps
  * Then train AMI/SWBD/WSJ nnet3 model)
* `multi_c` = your second experiment
  * Train GMM-HMM on SWBD
  * Then train SWBD/Tedlium nnet3 model
* ...

The `data` and `exp` directories for these variants can exist side-by-side: `multi_x` uses `data/multi_x` for training data directories and `exp/multi_x` for exp directories. This means you can easily train models on arbitrary combinations of whatever corpora you have on hand without overwriting previous work&mdash;simply create one recipe variant per experiment.

### Training partitions

Instead of having a few `train_*` directories (like `train_200k`, `train_50_shortest`), there is one such directory (or symlink) for each step during training, e.g.

```bash
$ ls -1 data/multi_a/
mono/
mono_ali/
tri1@
tri1_ali@
tri2@
tri2_ali/
tri3@
tri3_ali/
tri4@
# ...
```

The result is that the training script is much easier to read, since it basically boils down to

| Do ... | ... with the data in ... | ... and output the model to ... |
| --- | --- | --- |
| training | `data/multi_a/mono` | `exp/multi_a/mono` |
| alignment | `data/multi_a/mono_ali` | `exp/multi_a/mono_ali` |
| training | `data/multi_a/tri1` | `exp/multi_a/tri1` |
| training | `data/multi_a/tri2a` | `exp/multi_a/tri2b` |

What training data to use for each stage is specified by `local/make_partitions.sh`, which creates the `data/{MULTI}/{STAGE}` directories.

Again, this convention was chosen for its simplicity and extensibility.

### Full table

This table below lists all major structural differences between this recipe and the `fisher_swbd` recipe ([link](https://github.com/kaldi-asr/kaldi/tree/master/egs/fisher_swbd/s5/)).

| Description | Location in `fisher_swbd` | Location in this recipe | Example(s) in this recipe
| --- | --- | --- | --- |
| Config files | `conf` | `conf/{CORPUS}` | `conf/swbd` |
| Corpora-specific data directories for training | `data/train_swbd` or `data/train_fisher` | `data/{CORPUS}/train` | `data/tedlium/train` |
| Data directories for testing | `data/rt03` or `data/eval2000` | `data/{CORPUS}/test` | `data/tedlium/test` or `data/eval2000/test` |
| Data directories used during training (may be comprised of data from multiple corpora) |  `data/train_all` | `data/{MULTI}/{STAGE}` | `data/multi_a/tri2a` |
| Results |  `exp` | `exp/{MULTI}` | `exp/multi_a` |

## Scripts copied from other recipes

The files in `local/` that are prefixed with a database name (e.g. `wsj_*`) come
from those respective recipes. There is one exception: files that start with `fisher_`
or `swbd_` come from the `fisher_swbd` recipe.

Each script copied from another recipe has a header that explains 1) where the file
was copied from, 2) what revision it was copied from, and 3) what changes were made.
