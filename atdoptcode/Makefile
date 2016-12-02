CC = cc
OPTS = -g -Wall -pedantic -std=c99

# Static libraries to link to
ALGORITHMS_LIB = all_algorithms.a

# If a max, likely to find the library at /opt/local
# If linux, then at /usr/local
ifeq ($(USER), yangpan)
	INCLUDE = -I/opt/local/include
	LIBS = -L/N/soft/cle4/gsl/1.15/lib -lgsl -lgslcblas -lm
else
	INCLUDE = -I/usr/local/include
	LIBS = -L/usr/local/lib -lgsl -lgslcblas -lm 
endif

all: 
	make clean
	cd algorithms && make $(ALGORITHMS_LIB)
	make compare_algorithms

clean:
	rm -f *.o compare_algorithms
	cd algorithms && $(MAKE) clean

erfan:
	make clean
	cd algorithms && make $(ALGORITHMS_LIB)
	make compare_algorithms_erfan

#——————————————————————————————————————————————————————————————————————
# Rules to generate algorithms
#——————————————————————————————————————————————————————————————————————
algorithms/$(ALGORITHMS_LIB): algorithms/*.c algorithms/*.h
	cd algorithms && $(MAKE) $(ALGORITHMS_LIB)

utils.o: 
	$(CC) $(OPTS) $(INCLUDE) -c utils.c

mdp.o: 
	$(CC) $(OPTS) $(INCLUDE) -c mdp.c

model_mdp.o: mdp.o
	$(CC) $(OPTS) $(INCLUDE) -c model_mdp.c

boyan_mdp.o: mdp.o
	$(CC) $(OPTS) $(INCLUDE) -c boyan_mdp.c

trajectory_mdp.o: mdp.o
	$(CC) $(OPTS) $(INCLUDE) -c trajectory_mdp.c

experiment_utils.o: algorithms/$(ALGORITHMS_LIB) boyan_mdp.o trajectory_mdp.o utils.o
	$(CC) $(OPTS) $(INCLUDE) -c experiment_utils.c

# TODO: why cannot we compile into a static library? for now, putting individual .o files here
compare_algorithms: algorithms/$(ALGORITHMS_LIB) experiment_utils.o
	$(CC) $(OPTS) $(INCLUDE) -o compare_algorithms compare_algorithms.c mdp.o boyan_mdp.o experiment_utils.o trajectory_mdp.o utils.o algorithms/algorithm.o algorithms/linear_algorithms.o algorithms/sketch_algorithms.o  algorithms/matrix_algorithms.o algorithms/algorithm_utils.o $(LIBS)
	
compare_algorithms_erfan: algorithms/$(ALGORITHMS_LIB) experiment_utils.o
	$(CC) $(OPTS) $(INCLUDE) -o compare_algorithms_erfan compare_algorithms_erfan.c mdp.o boyan_mdp.o experiment_utils.o trajectory_mdp.o utils.o algorithms/algorithm.o algorithms/linear_algorithms.o algorithms/sketch_algorithms.o  algorithms/matrix_algorithms.o algorithms/algorithm_utils.o $(LIBS)
