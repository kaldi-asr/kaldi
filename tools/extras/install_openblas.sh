#!/bin/bash

# OpenBLAS is downloaded and built by tools/Makefile, but not automatically by
# its default 'all' target.
make -j openblas
