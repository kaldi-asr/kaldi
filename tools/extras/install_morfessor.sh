#!/bin/bash
# Copyright  2017  Atlas Guide (Author : Lucas Jo)
#
# Apache 2.0
#

echo "#### installing morfessor"
dirname=morfessor
if [ ! -d ./$dirname ]; then
  mkdir -p ./$dirname
  git clone https://github.com/aalto-speech/morfessor.git morfessor ||
    {
      if [ $? -ne 0 ]; then
        echo  >&2 "$0: Error git clone operation "
        echo  >&2 "  Failed in cloning the github repository (https://github.com/aalto-speech/morfessor.git)"
        echo  >&2 "  Download morfessor-2.0.1.tar.gz instead"
        wget http://morpho.aalto.fi/projects/morpho/morfessor-2.0.1.tar.gz
        tar -zxvf morfessor-2.0.1.tar.gz
        mv Morfessor-2.0.1/ $dirname
      fi
    }
fi

# local installation
site_packages_dir=$(python -m site --user-site | grep -oE "lib.*")
wd=`pwd`
wd=`readlink -f $wd || pwd`
export MORFESSOR="$wd/morfessor"
export PYTHONPATH="${PYTHONPATH:-}:$MORFESSOR/${site_packages_dir}"
(
cd $dirname
mkdir -p $site_packages_dir
python setup.py install --prefix `pwd` --record files.txt
)
wait

# env.sh setup
site_packages_dir=$(cd morfessor; find ./lib{,64} -type d -name site-packages | head -n 1)
(
  set +u
  [ ! -z "${MORFESSOR}" ] && \
    echo >&2 "morfessor variable is aleady defined. undefining..." && \
    unset MORFESSOR

  [ -f ./env.sh ] && . ./env.sh

  [ ! -z "${MORFESSOR}" ] && \
    echo >&2 "MORFESSOR config is already in env.sh" && exit

  wd=`pwd`
  wd=`readlink -f $wd || pwd`

  echo "export MORFESSOR=\"$wd/morfessor\""
  echo "export PATH=\"\$PATH:\${MORFESSOR}/bin\""
  echo "export PYTHONPATH=\"\${PYTHONPATH:-}:\$MORFESSOR/${site_packages_dir}\""
) >> env.sh

echo >&2 "installation of MORFESSOR finished successfully"
echo >&2 "please source tools/env.sh in your path.sh to enable it"
echo >&2 'when uninstall it, try: sudo rm $(cat ./morfessor/files.txt)'
