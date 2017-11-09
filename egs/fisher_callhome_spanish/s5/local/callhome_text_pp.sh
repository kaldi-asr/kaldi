#!/usr/bin/env bash
# Copyright 2014  Gaurav Kumar.   Apache 2.0

if [ $# -gt 0 ]; then
    sentence=$1
    echo $sentence | sed 's:{^[}]*}:[noise]:'
fi


