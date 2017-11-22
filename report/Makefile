LATEX=pdflatex  # or latex, xelatex, whichever you have
LATEXOPT=--shell-escape  # needed for \usepackage{minted}
NONSTOP=--interaction=nonstopmode

LATEXMK=latexmk
LATEXMKOPT=-pdf
CONTINUOUS=-pvc

MAIN=egpaper_final
SOURCES=$(MAIN).tex Makefile # sections/*tex
FIGURES := $(shell find figures/* images/* plots/* -type f)

# Default just builds it once (continuous mode disabled for latexmk).
all:	once

# Use make debug to view errors when latexmk is running in continuous mode.
continuous:	$(MAIN).pdf

.refresh:
	touch .refresh

$(MAIN).pdf: $(MAIN).tex .refresh $(SOURCES) $(FIGURES)
	$(LATEXMK) $(LATEXMKOPT) $(CONTINUOUS) -pdflatex="$(LATEX) $(LATEXOPT) $(NONSTOP) %O %S" $(MAIN)

force:
	touch .refresh
	rm $(MAIN).pdf
	$(LATEXMK) $(LATEXMKOPT) $(CONTINUOUS) -pdflatex="$(LATEX) $(LATEXOPT) %O %S" $(MAIN)

clean:
	$(LATEXMK) -C
	rm -f $(MAIN).pdfsync
#	rm -rf *~ *.tmp
	rm -rf _minted-paper/
	rm -f *.aux *.dvi *.log *.blg *.bbl *-blx.bib *.xml

once:
	$(LATEXMK) $(LATEXMKOPT) -pdflatex="$(LATEX) $(LATEXOPT) %O %S" $(MAIN)

debug:
	$(LATEX) $(LATEXOPT) $(MAIN)

.PHONY: clean force once all
