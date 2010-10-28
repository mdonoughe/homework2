CFLAGS = -Wall -g -O3 -DTURBO
LDFLAGS = -framework OpenGL -framework GLUT

all: main doc.pdf

doc.pdf: doc.tex
	xelatex doc.tex

main: main.o shaderbuilder.o nn.o

main.o: main.h shaderbuilder.h nn.h

nn.o: main.h nn.h

shaderbuilder.o: main.h shaderbuilder.h

clean:
	rm -f main *.o doc.aux doc.pdf doc.log

.PHONY: clean all
