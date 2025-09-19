#!/bin/bash

cd plot_files

# call python on first arg
python "$1.py"

# change all "../" to "../PlotNeuralNet/" in the tex file
sed -i 's|\.\./|./PlotNeuralNet/|g' "$1.tex"

cd ..

mkdir -p target

# build the tex file into a pdf
pdflatex -output-directory=./target "plot_files/$1.tex"
