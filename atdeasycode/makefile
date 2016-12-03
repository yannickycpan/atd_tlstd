INCLUDE = -I/usr/local/include
LIBS = -L/usr/local/lib -lgsl -lgslcblas -lm

OPTS = -Wall -pedantic -std=c99

all: 
	make clean
	make utils.o
	make domain_mdp.o
	make algorithms.o
	make comp_algo_domain

clean:
	rm -f *.o comp_algo_domain

utils.o: 
	cc $(OPTS) $(INCLUDE) -c utils.c

domain_mdp.o: utils.o
	cc $(OPTS) $(INCLUDE) -c domain_mdp.c

algorithms.o: utils.o domain_mdp.o
	cc $(OPTS) $(INCLUDE) -c algorithms.c

comp_algo_domain: utils.o domain_mdp.o algorithms.o
	cc $(OPTS) $(INCLUDE) -o comp_algo_energy comp_algo_energy.c algorithms.o utils.o domain_mdp.o $(LIBS)
