#!/usr/bin/env bash
#Copyright 2012 Cisco Systems; Matthias Paulik
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#https://www.apache.org/licenses/LICENSE-2.0
#
#THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
#WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
#MERCHANTABLITY OR NON-INFRINGEMENT.
#See the Apache 2 License for the specific language governing permissions and
#limitations under the License.
#
#This script attempts to install port audio, which is needed for the run-on
#decoding stuff. Portaudio enables the decoder to grab a live audio stream
#from the soundcard. I tested portaudio on Linux (RedHat and Suse Linux) and
#on MacOS 10.7. On Linux, it compiles out of the box. For MacOS 10.7,
#it is necessary to edit the Makefile (this script tries to do that).
#The script will remove all occurances of
#
# -Werror (occurs once in the Makefile)
# -arch i386 (occurs twice in the Makefile)
# -arch ppc (occurs twice in the Makefile)
# -isysroot /Developer/SDKs/MacOSX10.4u.sdk
#
#also, it seems that one has to uncomment the inclusion of AudioToolbox in
#include/pa_mac_core.h
#
#All this should make it compile fine for x86_64 under MacOS 10.7
#(always assuming that you installed XCode, wget and
#the Linux environment stuff on MacOS)

VERSION=v19_20111121

WGET=${WGET:-wget}

echo "****() Installing portaudio"

portaudio_tarball="pa_stable_${VERSION}.tgz"
portaudio_github_tarball="pa_stable_${VERSION}_r1788.tar.gz"

if [ ! -e $portaudio_tarball ]; then
    echo "Could not find portaudio tarball $portaudio_tarball locally, downloading it..."

    if [ -d "$DOWNLOAD_DIR" ]; then
        cp -p "$DOWNLOAD_DIR/$portaudio_tarball" .
    else
        if ! $WGET --version >&/dev/null; then
            echo "This script requires you to first install wget"
            echo "You can also just download pa_stable_$VERSION.tgz from"
            echo "http://www.portaudio.com/download.html)"
            exit 1;
        fi
	$WGET -nv -T 10 -t 3 -O $portaudio_tarball https://github.com/PortAudio/portaudio/archive/refs/tags/${portaudio_github_tarball} || \
	$WGET -nv -T 10 -t 3 -O $portaudio_tarball http://www.portaudio.com/archives/$portaudio_tarball || \
	rm ${portaudio_tarball}

    fi

    if [ ! -e $portaudio_tarball ]; then
        echo "Download of $portaudio_tarball - failed."
        echo "Aborting script. Please download and install port audio manually."
        exit 1;
    fi
fi

mkdir portaudio && tar -xzf $portaudio_tarball -C portaudio --strip-components 1 || exit 1

read -d '' pa_patch << "EOF"
--- portaudio/Makefile.in	2012-08-05 10:42:05.000000000 +0300
+++ portaudio/Makefile_mod.in	2012-08-05 10:41:54.000000000 +0300
@@ -193,6 +194,8 @@
 	for include in $(INCLUDES); do \\
 		$(INSTALL_DATA) -m 644 $(top_srcdir)/include/$$include $(DESTDIR)$(includedir)/$$include; \\
 	done
+	$(INSTALL_DATA) -m 644 $(top_srcdir)/src/common/pa_ringbuffer.h $(DESTDIR)$(includedir)/$$include;
+	$(INSTALL_DATA) -m 644 $(top_srcdir)/src/common/pa_memorybarrier.h $(DESTDIR)$(includedir)/$$include;
 	$(INSTALL) -d $(DESTDIR)$(libdir)/pkgconfig
 	$(INSTALL) -m 644 portaudio-2.0.pc $(DESTDIR)$(libdir)/pkgconfig/portaudio-2.0.pc
 	@echo ""

EOF

MACOS=`uname 2>/dev/null | grep Darwin`

cd portaudio

if [ -z "$MACOS" ]; then
    echo "Patching Makefile.in to include ring buffer functionality..."
    echo "${pa_patch}" | patch -p0 Makefile.in
fi

patch -p0  Makefile.in < ../extras/portaudio.patch
autoconf
./configure --prefix=`pwd`/install --with-pic
perl -i -pe 's:src/common/pa_ringbuffer.o:: if /^OTHER_OBJS\s*=/' Makefile

if [ "$MACOS" != "" ]; then
    echo "detected MacOS operating system ... trying to fix Makefile"
    mv Makefile Makefile.bck
    cat Makefile.bck | sed -e 's/\-isysroot\ \/Developer\/SDKs\/MacOSX10\.4u\.sdk//g' \
      -e 's/-Werror//g' -e 's/-arch i386//g' -e 's/-arch ppc64//g' -e 's/-arch ppc//g' \
      > Makefile
    mv include/pa_mac_core.h include/pa_mac_core.h.bck
    cat include/pa_mac_core.h.bck \
      | sed 's/\/\/\#include \<AudioToolbox\/AudioToolbox.h\>/#include \<AudioToolbox\/AudioToolbox.h\>/g' \
      > include/pa_mac_core.h
fi

make
make install

if [ "$MACOS" != "" ]; then
    cp src/common/pa_ringbuffer.h install/include/
fi

cd ..
