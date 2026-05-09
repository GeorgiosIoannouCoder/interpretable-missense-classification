.PHONY: all clean

all: final_report_team19_gi2100_vsj7589.pdf

final_report_team19_gi2100_vsj7589.pdf: main.tex references.bib ../figures/fig1_pipeline.pdf
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	bibtex main
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	mv main.pdf final_report_team19_gi2100_vsj7589.pdf

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.toc *.fdb_latexmk *.fls
