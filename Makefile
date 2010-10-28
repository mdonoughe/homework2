CFLAGS = -Wall -g -O3 -DTURBO -arch ppc -arch i386 -arch x86_64 -isysroot /Developer/SDKs/MacOSX10.5.sdk -mmacosx-version-min=10.5
LDFLAGS = -framework OpenGL -framework GLUT -arch ppc -arch i386 -arch x86_64 -Wl,-syslibroot,/Developer/SDKs/MacOSX10.5.sdk -mmacosx-version-min=10.5

all: homework2 doc.pdf

doc.pdf: doc.tex
	xelatex -o $<

homework2: main.o shaderbuilder.o nn.o
	$(CC) $(LDFLAGS) -o $@ $^

main.o: main.h shaderbuilder.h nn.h

nn.o: main.h nn.h

shaderbuilder.o: main.h shaderbuilder.h

clean:
	rm -f homework2 *.o doc.aux doc.pdf doc.log

.PHONY: clean all
