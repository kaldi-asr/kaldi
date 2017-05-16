documentationdir = @prefix@/doc

if DOC_COMPILATION

irstlm-manual.pdf: irstlm-manual.tex
	pdflatex irstlm-manual ; \
	bibtex irstlm-manual ; \
	pdflatex irstlm-manual ; \
	pdflatex irstlm-manual ; \
	rm *.aux *.log *.bbl *.blg

clean-local:
	rm -rf irstlm-manual.pdf

all: irstlm-manual.pdf
 
dist_documentation_SCRIPTS = irstlm-manual.pdf

endif

EXTRA_DIST = irstlm-manual.tex 


