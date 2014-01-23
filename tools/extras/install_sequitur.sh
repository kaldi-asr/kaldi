#!/bin/bash
set -u
set -e
g2p_archive=g2p-r1668.tar.gz


# Make sure we are in the tools/ directory.
if [ `basename $PWD` == extras ]; then
  cd ..
fi

! [ `basename $PWD` == tools ] && \
  echo "You must call this script from the tools/ directory" && exit 1;


[ ! -f $g2p_archive ] && wget http://www-i6.informatik.rwth-aachen.de/web/Software/$g2p_archive
tar xzf $g2p_archive
mv g2p sequitur


cd sequitur
patch  < ../sequitur.patch 
make
python setup.py install --prefix `pwd`

cd ../

(
set -x
echo "export G2P=`pwd`/sequitur"
echo "export PATH=\$PATH:\${G2P}/bin"
echo "_site_packages=\`readlink -f \${G2P}/lib/python*/site-packages\`"
echo "export PYTHONPATH=\$PYTHONPATH:\$_site_packages"
) >> env.sh
