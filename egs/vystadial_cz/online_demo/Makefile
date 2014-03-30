BEST_LINE=18
MODEL_PREFIX_URL=http://vystadial.ms.mff.cuni.cz/download/kaldi/src/pykaldi/pykaldi/binutils/
DATA_PREFIX_URL=http://vystadial.ms.mff.cuni.cz/download/kaldi/src/pykaldi/pykaldi/binutils/

# Czech language models 
LANG=cs
HCLG=models/HCLG_tri2b_bmmi.fst
AM=models/tri2b_bmmi.mdl
MAT=models/tri2b_bmmi.mat  # matrix trained in tri2b models 
WST=models/words.txt
MFCC=models/mfcc.conf
SILENCE=models/silence.csl

$(MFCC):
	wget $(MODEL_PREFIX_URL)/$@ -O $@

$(SILENCE):
	wget $(MODEL_PREFIX_URL)/$@ -O $@

$(HCLG):
	wget $(MODEL_PREFIX_URL)/$@ -O $@

$(AM):
	wget $(MODEL_PREFIX_URL)/$@ -O $@

$(MAT):
	wget $(MODEL_PREFIX_URL)/$@ -O $@

$(WST):
	wget $(MODEL_PREFIX_URL)/$@ -O $@

data/vystadial-sample-$(LANG).tar.gz:
	wget $(DATA_PREFIX_URL)/$@ -O $@

data/vystadial-sample-$(LANG)/README.rst: data/vystadial-sample-$(LANG).tar.gz
	tar xf  $< -C data
	touch $@

data/vystadial-sample-$(LANG)/test/input.scp: data/vystadial-sample-$(LANG)/README.rst
	for f in `dirname $@`/*.wav ; do echo "`basename $$f`" "`pwd`/$$f" ; done > $@
		
data/vystadial-sample-$(LANG)/test/input_best.scp: data/vystadial-sample-$(LANG)/test/input.scp
	sed -n '$(BEST_LINE)p' < $< > $@

build_scp: data/vystadial-sample-$(LANG)/test/input_best.scp data/vystadial-sample-$(LANG)/test/input.scp

download_models: $(SILENCE) $(WST) $(HCLG) $(AM) $(MAT) $(MFCC)

pyonline-recogniser: build_scp download_models
	bash run_pyonline-latgen-recogniser.sh

online-recogniser: build_scp download_models
	bash run_online-latgen-recogniser.sh

gmm-latgen-faster: build_scp download_models
	bash run_gmm-latgen-faster.sh

live: download_models
	bash run_live-demo.sh
	
clean:
	rm -rf *.pyc data/* models/* decode/*

.PHONY: live pyonline-recogniser online-recogniser gmm-latgen-faster clean
