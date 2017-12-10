#!/bin/bash -x

set -e
JOBS=8

{

echo "Starting build at "`date`

if [[ $1 == '--deploy' ]] ; then
    echo "Deployment build mode. CUDA will be disabled."
    cuda_opt="--use-cuda=no"
    deploy=true
else
    echo "Training build mode."
    deploy=false 
fi
   
cd tools
extras/check_dependencies.sh
make clean
make -j $JOBS
cd -

cd src
./configure --openblas-root=../tools/OpenBLAS/install ${cuda_opt}
make clean
make -j depend
make -j $JOBS
make -C online2_py
cd -

} > >(tee -a build_stdout.log) 2> >(tee -a build_stderr.log >&2)
