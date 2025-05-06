CC = mpicc
CFLAGS = -O3 -fopenmp -Wall
LDFLAGS = -lmetis -lm -lgvc -lcgraph
OBJECTS = main.o

all: main

main: $(OBJECTS)
	$(CC) $(CFLAGS) -o main $(OBJECTS) $(LDFLAGS)

main.o: main.c
	$(CC) $(CFLAGS) -c main.c

clean:
	rm -f main *.o output/*.png output_IST/*.png