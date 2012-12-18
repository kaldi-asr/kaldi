# SHELL += -x

all: sph2pipe atlas clapack irstlm_tgt sclite_tgt openfst_tgt

clean: openfst_cleaned sclite_cleaned  irstlm_cleaned

openfst_cleaned:
		$(MAKE) -C openfst-1.3.2 clean

irstlm_cleaned:
		$(MAKE) -C irstlm clean 

sclite_cleaned:
		$(MAKE) -C sctk-2.4.0 clean

distclean:
		rm -rf openfst-1.3.2/
		rm -rf sctk-2.4.0/
		rm -rf ATLAS/
		rm -rf CLAPACK_include/
		rm -rf irstlm/
		rm -rf sph2pipe_v2.5/
		rm -rf sph2pipe_v2.5.tar.gz
		rm -rf atlas3.8.3.tar.gz
		rm -rf sctk-2.4.0-20091110-0958.tar.bz2
		rm -rf openfst-1.3.2.tar.gz
		rm -f openfst

openfst_tgt: openfst_compiled

.PHONY: openfst_compiled
openfst_compiled: openfst-1.3.2/Makefile
		cd openfst-1.3.2/; \
		$(MAKE) install

openfst-1.3.2/Makefile: openfst-1.3.2/.patched
		cd openfst-1.3.2/; \
		./configure --prefix=`pwd` --enable-static --disable-shared --enable-far --enable-ngram-fsts

openfst-1.3.2/.patched: | openfst-1.3.2_tgt
		-cd openfst-1.3.2/src/include/fst; \
		patch -p0 -N < ../../../../openfst.patch;
		touch $@ 	

openfst-1.3.2_tgt: openfst-1.3.2.tar.gz
		tar xozf openfst-1.3.2.tar.gz
		-rm openfst
		-ln -s openfst-1.3.2 openfst 

openfst-1.3.2.tar.gz: 
		wget -T 10 -t 3 --no-check-certificate http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.3.2.tar.gz

sclite_tgt: sclite_compiled

.PHONY: sclite_compiled
sclite_compiled: sctk-2.4.0/.patched
		cd sctk-2.4.0; \
		$(MAKE) config && $(MAKE) all && $(MAKE) install && $(MAKE) doc

sctk-2.4.0/.patched: | sctk-2.4.0
		cd sctk-2.4.0/; \
		for x in src/asclite/core/recording.{h,cpp}; do \
			sed 's/Filter::Filter/::Filter/' $$x > tmpf; mv tmpf $$x; \
		done;
		touch $@ 
 
sctk-2.4.0: sctk-2.4.0-20091110-0958.tar.bz2
		tar xojf sctk-2.4.0-20091110-0958.tar.bz2

sctk-2.4.0-20091110-0958.tar.bz2:
		wget --no-check-certificate -T 10 -t 3 https://sourceforge.net/projects/kaldi/files/sctk-2.4.0-20091110-0958.tar.bz2

irstlm_tgt: irstlm_compiled

.PHONY: irstlm_compiled
irstlm_compiled: irstlm/Makefile
		cd irstlm/; \
		$(MAKE); $(MAKE) install	

irstlm/Makefile: irstlm/.patched
		cd irstlm; \
		(./regenerate-makefiles.sh || ./regenerate-makefiles.sh) && \
		./configure --prefix `pwd` 


irstlm/.patched: | irstlm
		-cd irstlm;\
		patch -N -p0 < ../interpolatedwrite-5.60.02.patch
		touch $@ 		

irstlm:
		svn -r 398 co https://irstlm.svn.sourceforge.net/svnroot/irstlm/trunk irstlm


clapack: clapack_download cblas_download

clapack_download: | CLAPACK_include/clapack.h CLAPACK_include/f2c.h

cblas_download: CLAPACK_include/cblas.h

CLAPACK_include/cblas.h: CLAPACK_include |  atlas
		cp ATLAS/include/cblas.h CLAPACK_include/cblas.h
		#(test -f $@; touch $@)

CLAPACK_include:
		mkdir CLAPACK_include

CLAPACK_include/clapack.h: CLAPACK_include
		cd CLAPACK_include;\
		wget -T 10 -t 3 http://www.netlib.org/clapack/clapack.h
		touch $@

CLAPACK_include/f2c.h: CLAPACK_include
		cd CLAPACK_include;\
		wget -T 10 -t 3 http://www.netlib.org/clapack/f2c.h
		touch $@

atlas: ATLAS/include/cblas.h

ATLAS/include/cblas.h: | atlas3.8.3.tar.gz
		tar xozf atlas3.8.3.tar.gz ATLAS/include

atlas3.8.3.tar.gz:
		wget -T 10 -t 3 http://sourceforge.net/projects/math-atlas/files/Stable/3.8.3/atlas3.8.3.tar.gz

sph2pipe: sph2pipe_compiled

sph2pipe_compiled: sph2pipe_v2.5/sph2pipe

sph2pipe_v2.5/sph2pipe: | sph2pipe_v2.5
		cd sph2pipe_v2.5/; \
		$(CC) -o sph2pipe  *.c -lm

sph2pipe_v2.5: sph2pipe_v2.5.tar.gz
		tar xzf sph2pipe_v2.5.tar.gz

sph2pipe_v2.5.tar.gz:
		wget -T 10 -t 3 http://merlin.fit.vutbr.cz/kaldi/sph2pipe_v2.5.tar.gz

