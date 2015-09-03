
all: paper.pdf	
	cp paper.pdf ~/desktop/2012_icassp_lattices.pdf

paper.pdf: paper.ps
	ps2pdf paper.ps paper.pdf


paper.ps: paper.dvi
	dvips -o paper.ps paper.dvi

paper.dvi: paper.tex  refs.bib
	latex paper -o paper.dvi
	bibtex paper






