#!/usr/bin/env bash

if [ $# -gt 0 ]; then
    sentence=$1
    echo $sentence | sed 's:{^[}]*}:[noise]:' 
fi


