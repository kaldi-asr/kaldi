BEST_LINE=16
MODEL_PREFIX_URL=http://vystadial.ms.mff.cuni.cz/download/pykaldi/src/pykaldi/pykaldi/binutils
# Czech language models 
LANG=cs
HCLGa=HCLG_tri2a.fst
AMa=tri2a.mdl
HCLGb=HCLG_tri2b.fst
AMb=tri2b.mdl
MATb=tri2b.mat
WST=words.txt
MFCC=mfcc.conf
SILENCE=silence.csl

$(MFCC):
	wget $(MODEL_PREFIX_URL)/$@

$(SILENCE):
	wget $(MODEL_PREFIX_URL)/$@

$(HCLGa):
	wget $(MODEL_PREFIX_URL)/$@

$(HCLGb):
	wget $(MODEL_PREFIX_URL)/$@

$(AMa):
	wget $(MODEL_PREFIX_URL)/$@

$(AMb):
	wget $(MODEL_PREFIX_URL)/$@

$(MATb):
	wget $(MODEL_PREFIX_URL)/$@

$(WST):
	wget $(MODEL_PREFIX_URL)/$@

vystadial-sample-$(LANG).tar.gz:
	wget http://vystadial.ms.mff.cuni.cz/download/vystadial-sample/vystadial-sample-$(LANG).tar.gz

vystadial-sample-$(LANG)/README.rst: vystadial-sample-$(LANG).tar.gz
	tar xf $<
	touch $@

export BUILD_INPUT_SCP_PY

vystadial-sample-$(LANG)/test/input.scp: vystadial-sample-$(LANG)/README.rst
	for f in `dirname $@`/*.wav ; do echo "`basename $$f`" "`pwd`/$$f" ; done > $@
		
vystadial-sample-$(LANG)/test/input_best.scp: vystadial-sample-$(LANG)/test/input.scp
	sed -n '$(BEST_LINE)p' < $< > $@

build_scp: vystadial-sample-$(LANG)/test/input_best.scp vystadial-sample-$(LANG)/test/input.scp

download_models: $(SILENCE) $(WST) $(HCLGa) $(AMa) $(MATb) $(AMb) $(HCLGb) $(MFCC)

pykaldi-latgen-faster: build_scp download_models
	bash run_pykaldi-latgen-faster-decoder.sh

gmm-latgen-faster: build_scp download_models
	bash run_gmm-latgen-faster.sh

live: download_models
	bash run_live-demo.sh
	
clean:
	rm -rf *.pyc $(HCLG) $(AM) $(MAT_LDA) $(WST) vystadial-sample-*

.PHONY: live pykaldi-latgen-faster gmm-latgen-faster clean
