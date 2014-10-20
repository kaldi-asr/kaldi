#!/usr/bin/env bash

# Fixes the CALLHOME stm files 
# Copyright 2014  Gaurav Kumar.   Apache 2.0

data_dir=$1

cat $data_dir/stm | awk '{$1=substr(tolower($1),0,length($1)-4);print;}' > $data_dir/stm_new
mv $data_dir/stm $data_dir/stm.bak
mv $data_dir/stm_new $data_dir/stm
