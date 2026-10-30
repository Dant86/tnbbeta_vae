# Latexmk configuration for this project

# Use pdflatex by default
$pdf_mode = 1;
$pdflatex = 'pdflatex -interaction=nonstopmode -synctex=1 %O %S';

# For bibtex (natbib)
$bibtex_use = 2;

# For biblatex, uncomment below and comment above:
# $biber = 'biber %O %S';
# $pdf_mode = 1;
# $pdflatex = 'pdflatex -interaction=nonstopmode -synctex=1 %O %S';

# Clean up these additional extensions
$clean_ext = 'acn acr alg glg glo gls glsdefs ist run.xml synctex.gz';

# Output directory (optional - uncomment to use)
# $out_dir = 'build';
