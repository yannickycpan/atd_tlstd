# TODO: could assume that this makefile assumes CC and CFLAGS are set,
# since the main Makefile will set them
CC = cc
OPTS = -g -Wall -pedantic -std=c99

# If a max, likely to find the library at /opt/local
# If linux, then at /usr/local
ifeq ($(USER),yangpan)
	INCLUDE = -I/opt/local/include
	LIBS = -L/N/soft/cle4/gsl/1.15/lib -lgsl -lgslcblas -lm
else
	INCLUDE = -I/usr/local/include
	LIBS = -L/usr/local/lib -lgsl -lgslcblas -lm
endif

ALGORITHMS_LIB = all_algorithms.a


clean:
	rm -f *.o

algorithm_utils.o:
	$(CC) $(OPTS) $(INCLUDE) -c algorithm_utils.c

algorithm.o: 
	$(CC) $(OPTS) $(INCLUDE) -c algorithm.c

linear_algorithms.o: algorithm_utils.o
	$(CC) $(OPTS) $(INCLUDE) -c linear_algorithms.c

matrix_algorithms.o: algorithm_utils.o
	$(CC) $(OPTS) $(INCLUDE) -c matrix_algorithms.c

sketch_algorithms.o: algorithm_utils.o matrix_algorithms.o
	$(CC) $(OPTS) $(INCLUDE) -c sketch_algorithms.c

$(ALGORITHMS_LIB): algorithm.o linear_algorithms.o sketch_algorithms.o matrix_algorithms.o 
	ar -cvq $(ALGORITHMS_LIB) *.o 

#$(CC) $(OPTS) $(INCLUDE) -o $(ALGORITHMS_LIB) algorithm.o linear_algorithms.o sketch_algorithms.o matrix_algorithms.o algorithm_utils.o $(LIBS)
