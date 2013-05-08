 
This directory contains various installation scripts for programs that are required by
certain example scripts in the egs/ directory.  Those example scripts will either
call, or instruct you to call, these scripts as necessary.
All these scripts should be run from one level up, i.e. in tools/: for example,

cd ../tools
extras/install_atlas.sh

The older scripts here, such as install_atlas.sh, will have a soft link from one
level up, in tools/, because various things still expect them to be there, but
in future, the installation scripts will only be here.
