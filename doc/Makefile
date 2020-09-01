all: pdf html

clean:
	rm -rf *.svg plots DOC.pdf DOC.html

tex:
	pandoc -F pandoc-plot -s DOC.md -o DOC.tex

pdf:
	pandoc -F pandoc-plot -s --katex DOC.md -o DOC.pdf

html:
	pandoc -L tikz.lua -F pandoc-plot -s --katex DOC.md -o DOC.html

epub:
	pandoc -t epub3 --webtex -L tikz.lua -F pandoc-plot -s DOC.md -o DOC.epub
