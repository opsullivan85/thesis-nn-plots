@REM call python on first arg
python plot_files\%1.py
@REM change all "../" to "../PlotNeuralNet/" in the tex file
powershell -Command "(Get-Content plot_files\%1.tex) -replace '\.\./', '../PlotNeuralNet/' | Set-Content plot_files\%1.tex"
@REM build the tex file into a pdf
pdflatex plot_files\%1.tex -output-directory=target