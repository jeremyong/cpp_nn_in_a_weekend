all: pdf html

clean:
	rm -rf *.svg plots DOC.pdf DOC.html

pdf:
	pandoc -F pandoc-plot -s --katex DOC.md -o DOC.pdf

html:
	pandoc -L tikz.lua -F pandoc-plot -s --katex DOC.md -o DOC.html
