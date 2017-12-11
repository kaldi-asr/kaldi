#!/bin/bash -x

set -e
JOBS=8

{

echo "Starting build at "`date`

if [[ $1 == '--deploy' ]] ; then
    echo "Deployment build mode. CUDA will be disabled. Static linking will be used."
    configure_opts="--use-cuda=no --static-fst=yes --static-math=yes"
    deploy=true
else
    echo "Training build mode. Dynamic linking will be used."
    configure_opts="--shared"
fi

cd tools
extras/check_dependencies.sh
make clean
make -j $JOBS
cd -

cd src
./configure --openblas-root=../tools/OpenBLAS/install ${configure_opts}
make clean
make -j depend
if [ ! $deploy ]; then
    make -j $JOBS
fi
make depend -C online2_py
make -j $JOBS -C online2_py
cd -

} > >(tee -a build_stdout.log) 2> >(tee -a build_stderr.log >&2)
