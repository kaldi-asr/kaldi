
all: kaldi-poster.pdf code-samples.ps script-toplevel.ps script-lowlevel.ps

kaldi-poster.pdf: kaldi-poster.tex beamerthemekaldi1.sty code-samples.pdf
	pdflatex kaldi-poster


code-samples.pdf: code-samples.tex
	pdflatex code-samples.tex

code-samples.ps: code-samples.pdf # only needed so we can use ghostview to find bounding box.
	pdf2ps code-samples.pdf code-samples.ps


script-toplevel.pdf: script-toplevel.tex
	pdflatex script-toplevel.tex

script-toplevel.ps: script-toplevel.pdf # only needed so we can use ghostview to find bounding box.
	pdf2ps script-toplevel.pdf script-toplevel.ps

script-lowlevel.pdf: script-lowlevel.tex
	pdflatex script-lowlevel.tex

script-lowlevel.ps: script-lowlevel.pdf # only needed so we can use ghostview to find bounding box.
	pdf2ps script-lowlevel.pdf script-lowlevel.ps



#latex code-samples.tex
# dvips code-samples.dvi -o code-samples.eps
# mv code-samples.eps tmp.eps
#  cat tmp.eps | awk '/BoundingBox/{ print "%%BoundingBox: 128 517 486 725"; next; } {print;}' \
   > code-samples.eps
# ps2pdf -dEPSCrop code-samples.eps code-samples.pdf

