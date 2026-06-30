Compile with uplatex + dvipdfmx:

uplatex -synctex=1 -interaction=nonstopmode -file-line-error spilled_energy_lecture_beamer.tex
uplatex -synctex=1 -interaction=nonstopmode -file-line-error spilled_energy_lecture_beamer.tex
dvipdfmx -o spilled_energy_lecture_beamer.pdf spilled_energy_lecture_beamer.dvi

Or with latexmk configured for uplatex/dvipdfmx.
