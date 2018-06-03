#
# Makefile for SRI-LM man pages
# 
# $Header: /home/srilm/CVS/srilm/man/Makefile,v 1.22 2011/12/22 22:53:45 stolcke Exp $
#

# try two ways of running groff
GROFF = (gnroff -c -Tascii -mgan 2>/dev/null || \
		groff -mtty-char -c -Tascii -man)

MAN1PAGES = \
	ngram \
	ngram-count \
	ngram-class \
	ngram-merge \
	disambig \
	nbest-lattice \
	nbest-optimize \
	nbest-mix \
	nbest-pron-score \
	segment \
	segment-nbest \
	hidden-ngram \
	anti-ngram \
	multi-ngram \
	lattice-tool \
	training-scripts \
	lm-scripts \
	ppl-scripts \
	pfsg-scripts \
	nbest-scripts \
	select-vocab \
	metadb

MAN3PAGES = \
	Prob \
	Vocab \
	LM \
	File

MAN5PAGES = \
	ngram-format \
	nbest-format \
	classes-format \
	pfsg-format \
	wlat-format

MAN7PAGES = \
	ngram-discount \
	srilm-faq

CATS = $(MAN1PAGES:%=cat1/%.1) \
	$(MAN3PAGES:%=cat3/%.3) \
	$(MAN5PAGES:%=cat5/%.5) \
	$(MAN7PAGES:%=cat7/%.7)

HTMLS = $(MAN1PAGES:%=html/%.1.html) \
	$(MAN3PAGES:%=html/%.3.html) \
	$(MAN5PAGES:%=html/%.5.html) \
	$(MAN7PAGES:%=html/%.7.html)

all:	cats htmls index

cats:	$(CATS)

index:	whatis windex

windex: $(CATS)
	rm -f windex
	scripts/makewhatis .

whatis:	$(CATS)
	rm -f whatis
	scripts/makewhatis.bsd .

htmls:	$(HTMLS)

cat1/%.1:	man1/%.1
	$(GROFF) < $< >$@
cat3/%.3:	man3/%.3
	$(GROFF) < $< >$@
cat5/%.5:	man5/%.5
	$(GROFF) < $< >$@
cat7/%.7:	man7/%.7
	$(GROFF) < $< >$@
	
html/%.1.html:	man1/%.1
	scripts/man2html $< >$@ 
html/%.3.html:	man3/%.3
	scripts/man2html $< >$@ 
html/%.5.html:	man5/%.5
	scripts/man2html $< >$@ 
html/%.7.html:	man7/%.7
	scripts/man2html $< >$@ 

clean:
	rm -f $(CATS) $(HTMLS)
