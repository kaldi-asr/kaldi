PAPER=icassp12

all: $(PAPER).pdf
	#cp $(PAPER).pdf ~/desktop/2012_icassp_semicont.pdf || true

ifeq ($(shell uname), Darwin)
$(PAPER).pdf: $(PAPER).tex $(PAPER).bbl
	pdflatex $(PAPER)
	open $(PAPER).pdf
else
$(PAPER).pdf: $(PAPER).tex $(PAPER).bbl
	pdflatex $(PAPER)

endif


$(PAPER).bbl: $(PAPER).tex refs-eig.bib refs.bib
	pdflatex $(PAPER)
	bibtex $(PAPER)
	pdflatex $(PAPER)

clean:
	-rm $(PAPER).aux $(PAPER).blg $(PAPER).bbl $(PAPER).log *~

.PHONY: clean
