# Copyright 2016    Johns Hopkins University (Dan Povey)
#           2016    Vijayaditya Peddinti
#           2016    Yiming Wang
# Apache 2.0.

"""This library has classes and methods to form neural network computation graphs,
in the nnet3 framework, using higher level abstractions called 'layers'
(e.g. sub-graphs like LSTMS ).

Note : We use the term 'layer' though the computation graph can have a highly
non-linear structure as, other terms such as nodes/components have already been
used in C++ codebase of nnet3.

This is basically a config parser module, where the configs have very concise
descriptions of a neural network.

This module has methods to convert the xconfigs into a configs interpretable by
nnet3 C++ library.

It generates three different configs:
 'init.config' : which is the config with the info necessary for computing
               the preconditioning matrix i.e., LDA transform
               e.g.
                 input-node name=input dim=40
                 input-node name=ivector dim=100
                 output-node name=output input=Append(Offset(input, -2), Offset(input, -1), input, Offset(input, 1), Offset(input, 2), ReplaceIndex(ivector, t, 0)) objective=linear

 'ref.config' : which is a version of the config file used to generate
                a model for getting left and right context (it doesn't read
                anything for the LDA-like transform and/or
                presoftmax-prior-scale components)

 'final.config' : which has the actual config used to initialize the model used
                 in training i.e, it has file paths for LDA transform and
                 other initialization files
"""


__all__ = ["utils", "layers", "parser"]
