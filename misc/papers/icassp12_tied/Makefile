PAPER=icassp12

all: $(PAPER).pdf

ifeq ($(shell uname), Darwin)
$(PAPER).pdf: $(PAPER).tex $(PAPER).bbl
	pdflatex $(PAPER)
	open $(PAPER).pdf
else
$(PAPER).pdf: $(PAPER).tex $(PAPER).bbl
	pdflatex $(PAPER)
	cp $(PAPER).pdf ~/desktop/2012_icassp_semicont.pdf
endif


$(PAPER).bbl: $(PAPER).tex refs-eig.bib refs.bib
	pdflatex $(PAPER)
	bibtex $(PAPER)
	pdflatex $(PAPER)

clean:
	-rm $(PAPER).aux $(PAPER).blg $(PAPER).bbl $(PAPER).log *~

.PHONY: clean
