PAPER=kaldi_asru

all: $(PAPER).pdf
	cp $(PAPER).pdf ~/desktop/2011_asru_kaldi.pdf

ifeq ($(shell uname), Darwin)
$(PAPER).pdf: $(PAPER).tex $(PAPER).bbl
	pdflatex $(PAPER)
	open $(PAPER).pdf
else
$(PAPER).pdf: $(PAPER).tex $(PAPER).bbl
	pdflatex $(PAPER)
endif


$(PAPER).bbl: $(PAPER).tex refs.bib
	pdflatex $(PAPER)
	bibtex $(PAPER)
	pdflatex $(PAPER)

clean:
	-rm $(PAPER).aux $(PAPER).blg $(PAPER).bbl $(PAPER).log *~

.PHONY: clean
