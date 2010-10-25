CFLAGS = -Wall -g
LDFLAGS = -framework OpenGL -framework GLUT

main: main.o shaderbuilder.o nn.o

main.o: main.h shaderbuilder.h nn.h

nn.o: main.h nn.h

shaderbuilder.o: main.h shaderbuilder.h

clean:
	rm -f main *.o

.PHONY: clean
