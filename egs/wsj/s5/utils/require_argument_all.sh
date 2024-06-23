#!/bin/bash

# To be sourced by another script

for i in $@; do
    . utils/require_argument.sh $i
done

