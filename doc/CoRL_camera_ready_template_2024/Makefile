MAIN := part_aware_pose_milestone

.PHONY: all
all:
	mkdir -p build
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build $(MAIN).tex
	bibtex build/$(MAIN)
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build $(MAIN).tex
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build $(MAIN).tex
	cp build/$(MAIN).pdf $(MAIN).pdf
